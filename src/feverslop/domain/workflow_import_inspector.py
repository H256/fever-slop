"""Deterministic, spec-driven inspection and validation of ComfyUI workflows.

One inspector serves every pipeline family: the family's boundary is supplied
as a :class:`WorkflowFamilySpec`. No per-family re-implementation is needed.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Mapping
from typing import Any

from feverslop.domain.workflow_import import (
    MAX_EDGES,
    MAX_NESTING,
    MAX_NODES,
    MAX_STRING_BYTES,
    MAX_WORKFLOW_BYTES,
    AnalysisIssue,
    BoundaryMapping,
    NodeCandidate,
    ValidationReport,
    WorkflowAnalysis,
    WorkflowFamilySpec,
)

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")
_WHITESPACE = re.compile(r"\s+")
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\)")


def _load_graph(graph: object) -> dict[str, Any]:
    if isinstance(graph, bytes):
        if len(graph) > MAX_WORKFLOW_BYTES:
            raise ValueError("workflow exceeds the 8 MiB limit")
        try:
            graph = json.loads(graph.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("workflow must be valid UTF-8 JSON") from exc
    elif isinstance(graph, str):
        raw = graph.encode("utf-8")
        if len(raw) > MAX_WORKFLOW_BYTES:
            raise ValueError("workflow exceeds the 8 MiB limit")
        try:
            graph = json.loads(graph)
        except json.JSONDecodeError as exc:
            raise ValueError("workflow must be valid JSON") from exc
    if not isinstance(graph, Mapping):
        raise TypeError("workflow must be a JSON object")
    normalized = {str(node_id): node for node_id, node in graph.items()}
    if len(normalized) != len(graph):
        raise ValueError("workflow contains duplicate node IDs after string normalization")
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_WORKFLOW_BYTES:
        raise ValueError("workflow exceeds the 8 MiB limit")
    if len(normalized) > MAX_NODES:
        raise ValueError("workflow exceeds the 2,000 node limit")
    _check_value_limits(normalized, depth=0)
    return normalized


def _check_value_limits(value: object, *, depth: int) -> None:
    if depth > MAX_NESTING:
        raise ValueError("workflow nesting exceeds depth 32")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise ValueError("workflow string exceeds the 64 KiB limit")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _check_value_limits(key, depth=depth + 1)
            _check_value_limits(child, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_value_limits(child, depth=depth + 1)


def _is_link(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and not isinstance(value[0], bool)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
        and value[1] >= 0
    )


def _graph_edges(
    graph: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]], list[tuple[str, str, str]]]:
    outgoing = {node_id: set() for node_id in graph}
    incoming = {node_id: set() for node_id in graph}
    links: list[tuple[str, str, str]] = []
    for target_id, node in graph.items():
        if not isinstance(node, Mapping) or not isinstance(node.get("inputs"), Mapping):
            continue
        for input_name, value in node["inputs"].items():
            if not _is_link(value):
                continue
            if len(links) >= MAX_EDGES:
                raise ValueError("workflow exceeds the 10,000 graph edge limit")
            source_id = str(value[0])
            links.append((source_id, target_id, str(input_name)))
            if source_id in graph:
                outgoing[source_id].add(target_id)
                incoming[target_id].add(source_id)
    return outgoing, incoming, links


def _ancestors(start: str, incoming: Mapping[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    pending = deque([start])
    while pending:
        node_id = pending.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        pending.extend(sorted(incoming.get(node_id, ())))
    return visited


def _clean_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = _WHITESPACE.sub(" ", _CONTROL_CHARACTERS.sub(" ", value)).strip()[:256]
    return "" if _ABSOLUTE_PATH.match(value) else value


def _title(node: object) -> str:
    if not isinstance(node, Mapping) or not isinstance(node.get("_meta"), Mapping):
        return ""
    value = node["_meta"].get("title")
    return _clean_name(value) if isinstance(value, str) else ""


def _output_metadata(class_type: str, object_info: Mapping[str, Any] | None) -> Mapping[str, Any]:
    value = object_info.get(class_type) if object_info else None
    return value if isinstance(value, Mapping) else {}


def _candidate(
    node_id: str,
    graph: Mapping[str, Any],
    *,
    object_info: Mapping[str, Any] | None,
    terminal: bool = False,
) -> NodeCandidate:
    node = graph[node_id]
    class_type = str(node.get("class_type") or "")
    metadata = _output_metadata(class_type, object_info)
    title = _title(node)
    output_types = metadata.get("output")
    return NodeCandidate(
        node_id=node_id,
        class_type=class_type,
        title=title,
        terminal=terminal,
        output_types=tuple(
            str(item) for item in output_types if isinstance(item, (str, int, float))
        )
        if isinstance(output_types, (list, tuple))
        else (),
    )


def _core_candidates(graph: Mapping[str, Any], spec: WorkflowFamilySpec) -> list[NodeCandidate]:
    candidates: list[NodeCandidate] = []
    for node_id, node in graph.items():
        if isinstance(node, Mapping) and node.get("class_type") in spec.core_class_types:
            candidates.append(_candidate(node_id, graph, object_info=None))
    candidates.sort(key=lambda c: c.node_id)
    return candidates


def _output_candidates(graph: Mapping[str, Any], spec: WorkflowFamilySpec) -> list[NodeCandidate]:
    candidates: list[NodeCandidate] = []
    for node_id, node in graph.items():
        if isinstance(node, Mapping) and node.get("class_type") in spec.output_class_types:
            candidates.append(_candidate(node_id, graph, object_info=None, terminal=True))
    candidates.sort(key=lambda c: c.node_id)
    return candidates


def _seed_node(graph: Mapping[str, Any], spec: WorkflowFamilySpec) -> tuple[str, str] | None:
    if spec.seed_class_type is None or spec.seed_input is None:
        return None
    for node_id, node in graph.items():
        if not isinstance(node, Mapping) or node.get("class_type") != spec.seed_class_type:
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, Mapping) and spec.seed_input in inputs:
            return node_id, spec.seed_input
    return None


def _required_inputs_satisfied(
    node: Mapping[str, Any], spec: WorkflowFamilySpec
) -> tuple[bool, list[str]]:
    inputs = node.get("inputs")
    if not isinstance(inputs, Mapping):
        return False, list(spec.required_inputs)
    missing = [name for name in spec.required_inputs if name not in inputs]
    return not missing, missing


def inspect_workflow(
    graph: object,
    *,
    spec: WorkflowFamilySpec,
    object_info: Mapping[str, Any] | None = None,
) -> WorkflowAnalysis:
    """Deterministically inspect a workflow against one family's boundary spec."""
    normalized = _load_graph(graph)
    _, incoming, _ = _graph_edges(normalized)

    core_candidates = _core_candidates(normalized, spec)
    output_candidates = _output_candidates(normalized, spec)
    issues: list[AnalysisIssue] = []

    if not core_candidates:
        issues.append(
            AnalysisIssue(
                code="missing_core",
                message=f"no node with a core class_type for pipeline {spec.pipeline!r}",
            )
        )
    if not output_candidates:
        issues.append(
            AnalysisIssue(
                code="missing_output",
                message=f"no output node for pipeline {spec.pipeline!r}",
            )
        )

    mapping: BoundaryMapping | None = None
    if core_candidates and output_candidates:
        core = core_candidates[0]
        output = output_candidates[0]
        core_node = normalized[core.node_id]
        satisfied, missing = _required_inputs_satisfied(core_node, spec)
        if not satisfied:
            issues.append(
                AnalysisIssue(
                    code="missing_required_inputs",
                    message=f"core node {core.node_id} is missing required inputs {missing}",
                    node_id=core.node_id,
                )
            )
        reachable = _ancestors(output.node_id, incoming)
        if core.node_id not in reachable:
            issues.append(
                AnalysisIssue(
                    code="core_not_upstream_of_output",
                    message=f"core node {core.node_id} is not upstream of output node {output.node_id}",
                    node_id=core.node_id,
                )
            )
        seed = _seed_node(normalized, spec)
        if spec.seed_class_type is not None and seed is None:
            issues.append(
                AnalysisIssue(
                    code="missing_seed",
                    message=f"no seed node of class_type {spec.seed_class_type!r}",
                )
            )
        if not issues:
            mapping = BoundaryMapping(
                core_node_id=core.node_id,
                core_class_type=core.class_type,
                required_inputs=spec.required_inputs,
                output_node_id=output.node_id,
                seed_node_id=seed[0] if seed else None,
                seed_input=seed[1] if seed else None,
            )

    if not core_candidates or not output_candidates:
        compatibility = "unsupported"
    elif mapping is None:
        compatibility = "needs_confirmation"
    else:
        compatibility = "auto_compatible"

    return WorkflowAnalysis(
        compatibility=compatibility,
        mapping=mapping,
        core_candidates=tuple(core_candidates),
        output_candidates=tuple(output_candidates),
        issues=tuple(issues),
    )


def validate_workflow(
    graph: object,
    *,
    spec: WorkflowFamilySpec,
    object_info: Mapping[str, Any] | None = None,
) -> ValidationReport:
    """Validate a workflow; a test-run is required before activation (see store)."""
    try:
        analysis = inspect_workflow(graph, spec=spec, object_info=object_info)
    except (ValueError, TypeError) as exc:
        return ValidationReport(
            valid=False,
            issues=(AnalysisIssue(code="invalid_graph", message=str(exc)),),
        )
    if analysis.compatibility != "auto_compatible":
        return ValidationReport(
            valid=False,
            issues=analysis.issues or (
                AnalysisIssue(
                    code="not_auto_compatible",
                    message=f"workflow is {analysis.compatibility}, not auto_compatible",
                ),
            ),
        )
    return ValidationReport(valid=True, issues=())
