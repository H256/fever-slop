from __future__ import annotations

import unittest

from feverslop.domain.ensemble import (
    EnsembleConfig,
    load_ensembles,
    validate_ensemble_cast,
)


class EnsembleConfigTest(unittest.TestCase):
    def test_from_dict_full(self) -> None:
        ensemble = EnsembleConfig.from_dict(
            {
                "id": "main_band",
                "members": [
                    {"actor_id": "singer_01", "role": "singer"},
                    {"actor_id": "guitarist_01", "role": "guitarist"},
                ],
                "required_scene_types": ["performance"],
            }
        )
        self.assertEqual(ensemble.id, "main_band")
        self.assertEqual(
            ensemble.member_actor_ids, ("singer_01", "guitarist_01")
        )
        self.assertEqual(ensemble.required_scene_types, ("performance",))

    def test_from_dict_requires_id(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleConfig.from_dict({"members": []})

    def test_from_dict_requires_member_actor_id(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleConfig.from_dict(
                {"id": "band", "members": [{"role": "singer"}]}
            )

    def test_from_dict_rejects_duplicate_member(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleConfig.from_dict(
                {
                    "id": "band",
                    "members": [
                        {"actor_id": "a"},
                        {"actor_id": "a"},
                    ],
                }
            )

    def test_from_dict_rejects_non_list_members(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleConfig.from_dict({"id": "band", "members": "a"})

    def test_requires_all_members_is_case_insensitive(self) -> None:
        ensemble = EnsembleConfig.from_dict(
            {"id": "band", "members": [{"actor_id": "a"}], "required_scene_types": ["Performance"]}
        )
        self.assertTrue(ensemble.requires_all_members("performance"))
        self.assertTrue(ensemble.requires_all_members("PERFORMANCE"))
        self.assertFalse(ensemble.requires_all_members("narrative"))

    def test_load_ensembles_empty_and_duplicate(self) -> None:
        self.assertEqual(load_ensembles(None), ())
        self.assertEqual(load_ensembles([]), ())
        with self.assertRaises(ValueError):
            load_ensembles(
                [
                    {"id": "band", "members": [{"actor_id": "a"}]},
                    {"id": "band", "members": [{"actor_id": "b"}]},
                ]
            )


class ValidateEnsembleCastTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ensemble = EnsembleConfig.from_dict(
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

    def test_performance_missing_members_reported(self) -> None:
        missing = validate_ensemble_cast(
            self.ensemble,
            scene_actor_ids=["singer_01"],
            scene_type="performance",
        )
        self.assertEqual(missing, ("guitarist_01", "drummer_01"))

    def test_performance_all_present_is_empty(self) -> None:
        missing = validate_ensemble_cast(
            self.ensemble,
            scene_actor_ids=["singer_01", "guitarist_01", "drummer_01"],
            scene_type="performance",
        )
        self.assertEqual(missing, ())

    def test_narrative_subset_is_allowed(self) -> None:
        missing = validate_ensemble_cast(
            self.ensemble,
            scene_actor_ids=["singer_01"],
            scene_type="narrative",
        )
        self.assertEqual(missing, ())

    def test_narrative_empty_cast_is_allowed(self) -> None:
        missing = validate_ensemble_cast(
            self.ensemble,
            scene_actor_ids=[],
            scene_type="narrative",
        )
        self.assertEqual(missing, ())

    def test_empty_cast_on_performance_reports_all(self) -> None:
        missing = validate_ensemble_cast(
            self.ensemble,
            scene_actor_ids=[],
            scene_type="performance",
        )
        self.assertEqual(
            missing, ("singer_01", "guitarist_01", "drummer_01")
        )


if __name__ == "__main__":
    unittest.main()
