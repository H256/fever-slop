"""Semantic audio-delivery contract declared by MiniMax H3 workflow profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from functools import lru_cache
from pathlib import Path

from feverslop.domain.audio_timing_contract import AudioTimingWindow
from feverslop.domain.artifact_hash import sha256_file


_AUDIO_LATENT_POLICIES = frozenset({"preserve_original_av_audio_latent"})


@dataclass(frozen=True)
class H3AudioDelivery:
    """How a selected H3 workflow carries supplied audio into a render.

    This is deliberately derived from the workflow sidecar rather than audio
    filenames: a full mix can be a generation condition, an output copy, an
    audience-only score, or none of these depending on the workflow graph.
    """

    audio_policy: str = "not_applicable"
    conditions_generation: bool = False
    copies_to_output: bool = False
    is_audience_only_music: bool = False
    workflow_profile: str | None = None
    workflow_hash: str | None = None
    profile_hash: str | None = None
    workflow_path: str | None = None
    conditioning_source: str = "selected_performer"

    def to_context(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_context(cls, value: object) -> "H3AudioDelivery":
        if not isinstance(value, dict):
            return cls()
        return cls(
            workflow_hash=value.get("workflow_hash"),
            profile_hash=value.get("profile_hash"),
            workflow_path=str(value["workflow_path"]) if value.get("workflow_path") else None,
            conditioning_source=str(value.get("conditioning_source") or "selected_performer"),
            audio_policy=str(value.get("audio_policy") or "not_applicable"),
            conditions_generation=value.get("conditions_generation") is True,
            copies_to_output=value.get("copies_to_output") is True,
            is_audience_only_music=value.get("is_audience_only_music") is True,
            workflow_profile=(
                str(value["workflow_profile"])
                if value.get("workflow_profile") else None
            ),
        )


def load_h3_audio_delivery(workflow_path: str | Path | None) -> H3AudioDelivery:
    """Read a workflow's optional ``.profile.json`` audio delivery semantics.

    Missing or malformed sidecars never invent audio behavior. The renderer
    remains the authority for whether an audio reference is attached.
    """
    if workflow_path is None:
        return H3AudioDelivery()
    path = Path(workflow_path)
    profile_path = path.with_suffix(".profile.json")
    fingerprints = {"workflow_hash": _source_hash(path), "profile_hash": _source_hash(profile_path)}
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return H3AudioDelivery(workflow_path=str(path), **fingerprints)
    if not isinstance(raw, dict):
        return H3AudioDelivery(workflow_path=str(path), **fingerprints)
    policy = str(raw.get("audio_policy") or "not_applicable").strip().casefold()
    preserves_audio_latent = raw.get("preserve_audio_latent") is True
    has_audio_latent_topology = "#RECOMBINE_AV" in {
        str(item).strip().upper() for item in raw.get("topology") or ()
    }
    conditions_generation = (
        policy in _AUDIO_LATENT_POLICIES
        and preserves_audio_latent
        and has_audio_latent_topology
    )
    return H3AudioDelivery(
        **fingerprints,
        workflow_path=str(path),
        conditioning_source=str(raw.get("conditioning_source") or "selected_performer").strip().casefold(),
        audio_policy=policy,
        conditions_generation=conditions_generation,
        copies_to_output=conditions_generation,
        is_audience_only_music=False,
        workflow_profile=str(profile_path),
    )


class H3AudioContractError(ValueError):
    """A source declaration cannot be established from the workflow graph."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f'{code}: {message}')


def _audio_loader(workflow: dict, edge: object) -> tuple[str, str | None]:
    """Trace only the supported lossless loader/trim topology."""
    seen = set()
    trim_id = None
    while isinstance(edge, (list, tuple)) and len(edge) == 2 and edge[1] == 0:
        node_id = str(edge[0])
        if node_id in seen:
            break
        seen.add(node_id)
        node = workflow.get(node_id, {})
        if node.get('class_type') == 'LoadAudio':
            return node_id, trim_id
        if node.get('class_type') != 'TrimAudioDuration' or trim_id is not None:
            break
        trim_id = node_id
        edge = node.get('inputs', {}).get('audio')
    raise H3AudioContractError('h3_audio_unknown_topology', 'Audio input must trace through at most one TrimAudioDuration to LoadAudio.')


def _validate_latent_policy(delivery: H3AudioDelivery, workflow: dict) -> None:
    if delivery.audio_policy not in _AUDIO_LATENT_POLICIES:
        return
    recombiners = [node for node in workflow.values() if node.get('_meta', {}).get('title') == '#RECOMBINE_AV']
    for node in recombiners:
        edge = node.get('inputs', {}).get('audio_latent')
        if node.get('class_type') == 'LTXVConcatAVLatent' and isinstance(edge, list) and len(edge) == 2 and edge[1] == 1:
            if workflow.get(str(edge[0]), {}).get('class_type') == 'LTXVSeparateAVLatent':
                return
    raise H3AudioContractError('h3_audio_profile_contradiction', 'Preserved AV audio latent policy does not match the graph.')


def resolve_h3_audio_sources(
    delivery: H3AudioDelivery,
    references: list[dict],
    window: AudioTimingWindow | None,
    *,
    workflow: dict | None = None,
) -> list[dict]:
    """Resolve expected source roles after numbered reference-slot patching.

    Reference identity comes from selected records; node titles only identify the
    renderer's numbered patch anchors, never the semantic identity of a source.
    """
    if workflow is None:
        if not delivery.workflow_path:
            if delivery.conditioning_source != 'selected_performer':
                raise H3AudioContractError('h3_audio_unknown_topology', 'An explicit source requires a workflow graph.')
            workflow = {}
        else:
            try:
                workflow = json.loads(Path(delivery.workflow_path).read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as exc:
                raise H3AudioContractError('h3_audio_unknown_topology', 'Cannot read the selected audio workflow.') from exc
    _validate_latent_policy(delivery, workflow)
    anchors = {node.get('_meta', {}).get('title'): str(node_id) for node_id, node in workflow.items()}
    records = []
    for index, ref in enumerate(references):
        source = str(ref.get('source') or ref.get('source_path') or '')
        if not source:
            raise H3AudioContractError('h3_audio_missing_source', 'Selected audio reference has no source path.')
        path = Path(source)
        records.append({
            'source_path': source, 'source_hash': _source_hash(path),
            'name': str(ref.get('name') or ''), 'label': str(ref.get('label') or ''),
            'roles': ['reference'], 'node_id': anchors.get(f'#AUDIO_{index + 1}'), 'input_name': 'audio',
            'reference_index': index,
            'bindings': [{'role': 'reference', 'node_id': str(node_id), 'input_name': f'ref_audios.ref_audio_{index}'}
                         for node_id, node in workflow.items() if node.get('class_type') == 'MiniMaxH3ReferenceToVideo'],
            'audio_timing_window': asdict(window) if window else None,
        })
    guides = [(str(node_id), node) for node_id, node in workflow.items() if node.get('class_type') == 'MiniMaxH3AddGuide' and 'audio' in node.get('inputs', {})]
    explicit = delivery.conditioning_source != 'selected_performer'
    if explicit and not guides:
        raise H3AudioContractError('h3_audio_profile_contradiction', 'Explicit conditioning source requires an audio guide input.')
    for node_id, guide in guides:
        loader_id, _ = _audio_loader(workflow, guide['inputs']['audio'])
        if explicit:
            candidates = [r for r in records if r['name'].casefold() == delivery.conditioning_source]
        else:
            candidates = [r for r in records if r['node_id'] == loader_id]
        if len(candidates) != 1:
            raise H3AudioContractError('h3_audio_missing_source', 'Guide conditioning source must identify exactly one selected reference.')
        record = candidates[0]
        if 'conditioning' not in record['roles']:
            record['roles'].append('conditioning')
        record['bindings'].append({'role': 'conditioning', 'node_id': node_id, 'input_name': 'audio'})
    for node_id, node in workflow.items():
        if node.get('class_type') not in {'VHS_VideoCombine', 'CreateVideo'}:
            continue
        edge = node.get('inputs', {}).get('audio')
        if not isinstance(edge, list) or not edge:
            continue
        # Decoded generated audio is not evidence that an external file is copied.
        if workflow.get(str(edge[0]), {}).get('class_type') not in {'LoadAudio', 'TrimAudioDuration'}:
            continue
        loader_id, _ = _audio_loader(workflow, edge)
        candidates = [r for r in records if r['node_id'] == loader_id]
        if len(candidates) != 1:
            raise H3AudioContractError('h3_audio_missing_source', 'Output audio must identify a selected source.')
        record = candidates[0]
        if 'output_copy' not in record['roles']:
            record['roles'].append('output_copy')
        record['bindings'].append({'role': 'output_copy', 'node_id': str(node_id), 'input_name': 'audio'})
    return records


def apply_h3_audio_sources(workflow: dict, sources: list[dict]) -> None:
    """Wire declared guide sources to their already patched reference slots."""
    anchors = {node.get('_meta', {}).get('title'): str(node_id) for node_id, node in workflow.items()}
    for source in sources:
        index = source['reference_index'] + 1
        target = anchors.get(f'#TRIM_AUDIO_{index}') or anchors.get(f'#AUDIO_{index}')
        for binding in source['bindings']:
            if binding['role'] == 'conditioning':
                if target is None:
                    raise H3AudioContractError('h3_audio_missing_source', 'Selected conditioning slot is absent.')
                workflow[binding['node_id']]['inputs'][binding['input_name']] = [target, 0]


def validate_h3_audio_sources(
    workflow: dict, sources: list[dict], uploaded_names: dict[str, str], *, delivery: H3AudioDelivery | None = None,
) -> None:
    """Verify actual guide/reference edges, uploaded source identity and timing."""
    if delivery is not None:
        _validate_latent_policy(delivery, workflow)
    for source in sources:
        for binding in source['bindings']:
            edge = workflow.get(binding['node_id'], {}).get('inputs', {}).get(binding['input_name'])
            loader_id, trim_id = _audio_loader(workflow, edge)
            if workflow[loader_id].get('inputs', {}).get('audio') != uploaded_names.get(source['source_path']):
                raise H3AudioContractError('h3_audio_source_mismatch', 'Patched audio loader differs from its declared source.')
            window = source.get('audio_timing_window')
            if window:
                inputs = workflow.get(trim_id, {}).get('inputs', {})
                if inputs.get('start_index') != window['start_seconds'] or inputs.get('duration') != round(window['end_seconds'] - window['start_seconds'], 6):
                    raise H3AudioContractError('h3_audio_timing_mismatch', 'Patched audio trim differs from its declared window.')


def h3_audio_timing_window(scene: dict, duration_seconds: float | None, *, fps: int = 24) -> AudioTimingWindow | None:
    """Use the renderer's absolute frame-aligned interval, including its anchor."""
    if duration_seconds is None:
        return None
    start = float(scene.get('abs_start_seconds', 0.0) or 0.0)
    end = float(scene.get('abs_end_seconds', start + float(duration_seconds)))
    return AudioTimingWindow((round(start * fps) - int(scene.get('anchor_frames') or 0)) / fps, round(end * fps) / fps)


@lru_cache(maxsize=128)
def _cached_source_hash(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    return sha256_file(Path(path))


def _source_hash(path: Path) -> str | None:
    try:
        stat = path.stat()
        if not path.is_file():
            return None
    except OSError:
        return None
    return _cached_source_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
