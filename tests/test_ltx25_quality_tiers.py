import json
import unittest
from pathlib import Path


WORKFLOW_ROOT = Path("workflows/video/ltx_25")
MODES = ("i2v", "msr", "t2v", "ingredients")
TIERS = ("draft", "standard", "final")


def _parse_sigmas(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def _sigma_nodes(payload: dict) -> dict[str, list[float]]:
    """Map node id -> sigma schedule for every ManualSigmas node."""
    nodes = {}
    for node_id, node in payload.items():
        if node.get("class_type") == "ManualSigmas":
            nodes[node_id] = _parse_sigmas(node["inputs"]["sigmas"])
    return nodes


class LTX25QualityTierTests(unittest.TestCase):
    """The draft/standard/final tier workflows must differ by sampling budget.

    Regression for issue #1304: the tier files were byte-identical, so selecting
    a higher quality tier produced the same (draft-quality) output. The tier
    differentiation is carried by the ManualSigmas sampling schedule; the rest of
    the graph is identical across tiers.
    """

    def test_tier_files_are_distinct_per_mode(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                payloads = {
                    tier: json.loads(
                        (WORKFLOW_ROOT / mode / f"{mode}_{tier}.json").read_text(
                            encoding="utf-8-sig"
                        )
                    )
                    for tier in TIERS
                }
                for a in TIERS:
                    for b in TIERS:
                        if a < b:
                            self.assertNotEqual(
                                payloads[a],
                                payloads[b],
                                f"{mode}: {a} and {b} are byte-identical",
                            )

    def test_tier_sampling_budget_increases_with_quality(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                schedules = {
                    tier: _sigma_nodes(
                        json.loads(
                            (WORKFLOW_ROOT / mode / f"{mode}_{tier}.json").read_text(
                                encoding="utf-8-sig"
                            )
                        )
                    )
                    for tier in TIERS
                }
                # Each tier exposes the same two-pass topology (two ManualSigmas).
                for tier in TIERS:
                    self.assertEqual(
                        len(schedules[tier]),
                        2,
                        f"{mode}/{tier}: expected two ManualSigmas nodes",
                    )
                # Identify the two passes by their starting sigma: pass 1 begins at
                # 1.0, pass 2 continues from pass 1's end (0.909375 in the draft).
                def pass_len(tier: str, start: float) -> int:
                    matches = [
                        len(v)
                        for v in schedules[tier].values()
                        if abs(v[0] - start) < 1e-9
                    ]
                    self.assertEqual(len(matches), 1, f"{mode}/{tier}: expected one pass starting at {start}")
                    return matches[0]

                for start in {round(s[0], 6) for s in schedules["draft"].values()}:
                    d = pass_len("draft", start)
                    s = pass_len("standard", start)
                    f = pass_len("final", start)
                    self.assertGreater(s, d, f"{mode}: standard pass not more sampled than draft")
                    self.assertGreater(f, s, f"{mode}: final pass not more sampled than standard")

    def test_tier_schedules_are_valid(self):
        for mode in MODES:
            for tier in TIERS:
                with self.subTest(mode=mode, tier=tier):
                    payload = json.loads(
                        (WORKFLOW_ROOT / mode / f"{mode}_{tier}.json").read_text(encoding="utf-8-sig")
                    )
                    for node_id, schedule in _sigma_nodes(payload).items():
                        self.assertEqual(schedule[-1], 0.0, f"{mode}/{tier} node {node_id}: schedule must end at 0.0")
                        self.assertTrue(
                            all(schedule[i] >= schedule[i + 1] for i in range(len(schedule) - 1)),
                            f"{mode}/{tier} node {node_id}: schedule must be monotonically decreasing",
                        )
                        self.assertTrue(0.0 <= schedule[0] <= 1.0, f"{mode}/{tier} node {node_id}: first sigma out of range")
