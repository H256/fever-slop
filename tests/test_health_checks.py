import unittest
from unittest.mock import MagicMock

from feverslop.adapters.health_checks import ServiceHealthChecker, build_service_probes


class HealthCheckTests(unittest.TestCase):
    def test_classifies_failures_and_calls_injected_alert(self):
        alerts = []
        checker = ServiceHealthChecker(
            {"comfyui": lambda: (_ for _ in ()).throw(ConnectionError("offline")),
             "llm": lambda: (_ for _ in ()).throw(PermissionError("denied"))},
            alert=alerts.append,
        )
        results = checker.check()
        self.assertEqual(["connectivity", "authentication"], [item.category for item in results])
        self.assertEqual(results, alerts)

    def test_healthy_probe_does_not_alert(self):
        alerts = []
        self.assertTrue(ServiceHealthChecker({"llm": lambda: True}, alerts.append).check()[0].healthy)
        self.assertEqual([], alerts)

    def test_build_service_probes_connects_standard_clients(self):
        comfyui = MagicMock()
        llm = MagicMock()
        probes = build_service_probes(comfyui=comfyui, llm=llm)

        self.assertEqual({"comfyui", "llm"}, set(probes))
        probes["comfyui"]()
        probes["llm"]()
        comfyui.health_check.assert_called_once_with()
        llm.health_check.assert_called_once_with()
