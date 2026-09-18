import unittest

from feverslop.application.visual_consistency_preflight import (
    preflight_visual_consistency,
)
from feverslop.domain.ensemble import EnsembleConfig
from feverslop.domain.visual_consistency import PreflightMode, ReferenceAnchor
from feverslop.ports.visual_consistency import ReferenceManifestSnapshot


def _anchor(kind: str, semantic_id: str) -> ReferenceAnchor:
    return ReferenceAnchor(
        id=semantic_id,
        kind=kind,
        look_id="default",
        asset_role="identity-reference" if kind == "actor" else "environment-reference",
        asset_sha256="a" * 64,
        prompt_anchor=f"{kind} {semantic_id}",
    )


def _snapshot() -> ReferenceManifestSnapshot:
    return ReferenceManifestSnapshot(
        actors={
            ("singer_01", "default"): _anchor("actor", "singer_01"),
            ("guitarist_01", "default"): _anchor("actor", "guitarist_01"),
            ("drummer_01", "default"): _anchor("actor", "drummer_01"),
        },
        locations={},
        revision="revision",
    )


def _ensemble() -> EnsembleConfig:
    return EnsembleConfig.from_dict(
        {
            "id": "main_band",
            "members": [
                {"actor_id": "singer_01"},
                {"actor_id": "guitarist_01"},
                {"actor_id": "drummer_01"},
            ],
            "required_scene_types": ["performance"],
        }
    )


def _scene(actor_ids: list[str], scene_type: str) -> dict:
    return {
        "scene": 1,
        "references": {
            "actor_ids": actor_ids,
            "actor_msr_paths": [f"role_{i}" for i in range(len(actor_ids))],
        },
        "metadata": {"type": scene_type},
    }


class EnsemblePreflightTest(unittest.TestCase):
    def test_performance_missing_member_blocks_in_strict(self) -> None:
        result = preflight_visual_consistency(
            [_scene(["singer_01"], "performance")],
            _snapshot(),
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode=PreflightMode.STRICT,
            ensembles=(_ensemble(),),
        )
        self.assertFalse(result.renderable)
        codes = [issue.code for issue in result.issues]
        self.assertIn("ensemble_incomplete", codes)
        self.assertEqual("error", [i.severity for i in result.issues if i.code == "ensemble_incomplete"][0])

    def test_performance_all_members_present_is_renderable(self) -> None:
        result = preflight_visual_consistency(
            [_scene(["singer_01", "guitarist_01", "drummer_01"], "performance")],
            _snapshot(),
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode=PreflightMode.STRICT,
            ensembles=(_ensemble(),),
        )
        self.assertTrue(result.renderable)
        self.assertNotIn(
            "ensemble_incomplete", [issue.code for issue in result.issues]
        )

    def test_narrative_subset_is_renderable(self) -> None:
        result = preflight_visual_consistency(
            [_scene(["singer_01"], "narrative")],
            _snapshot(),
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode=PreflightMode.STRICT,
            ensembles=(_ensemble(),),
        )
        self.assertTrue(result.renderable)
        self.assertNotIn(
            "ensemble_incomplete", [issue.code for issue in result.issues]
        )

    def test_no_ensembles_leaves_behavior_unchanged(self) -> None:
        result = preflight_visual_consistency(
            [_scene(["singer_01"], "performance")],
            _snapshot(),
            mode="msr",
            workflow_profile="msr-default",
            preflight_mode=PreflightMode.STRICT,
        )
        self.assertTrue(result.renderable)


if __name__ == "__main__":
    unittest.main()
