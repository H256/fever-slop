"""Dependency-free external service health probes with injectable alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class HealthResult:
    service: str
    healthy: bool
    category: str
    detail: str = ""


class ServiceHealthChecker:
    def __init__(self, probes: dict[str, Callable[[], object]], alert: Callable[[HealthResult], None] | None = None):
        self.probes = dict(probes)
        self.alert = alert

    def check(self) -> list[HealthResult]:
        results = []
        for service, probe in sorted(self.probes.items()):
            try:
                value = probe()
                healthy = value is not False
                result = HealthResult(service, healthy, "ok" if healthy else "failure")
            except PermissionError as exc:
                result = HealthResult(service, False, "authentication", str(exc))
            except (ConnectionError, TimeoutError) as exc:
                result = HealthResult(service, False, "connectivity", str(exc))
            except Exception as exc:  # probes are external boundaries
                result = HealthResult(service, False, "configuration", str(exc))
            results.append(result)
            if not result.healthy and self.alert is not None:
                self.alert(result)
        return results


def build_service_probes(*, comfyui: object | None = None, llm: object | None = None) -> dict[str, Callable[[], object]]:
    """Build standard non-mutating probes for configured external clients."""
    probes: dict[str, Callable[[], object]] = {}
    if comfyui is not None:
        probes["comfyui"] = getattr(comfyui, "health_check")
    if llm is not None:
        probes["llm"] = getattr(llm, "health_check")
    return probes
