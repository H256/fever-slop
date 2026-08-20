import unittest

from feverslop.domain.reference_contracts import render_reference_contract


class ReferenceContractTests(unittest.TestCase):
    def test_generic_contract_allows_ambient_population_but_protects_named_subjects(self):
        contract = render_reference_contract(
            [
                {"label": "<Picture 1>", "kind": "picture", "role": "subject", "name": "Ava"},
                {"label": "<Picture 2>", "kind": "picture", "role": "environment", "name": "Crowded Tavern"},
            ]
        )

        self.assertIn("exactly one persistent physical individual", contract)
        self.assertIn("Ambient background people may exist as anonymous extras", contract)
        self.assertIn("must not replace, duplicate, or impersonate", contract)
        self.assertNotIn("stage", contract.lower())
        self.assertNotIn("microphone", contract.lower())

    def test_live_concert_profile_adds_stage_and_role_bindings_only_when_selected(self):
        contract = render_reference_contract(
            [
                {"label": "<Picture 1>", "kind": "picture", "role": "subject", "name": "Singer"},
                {"label": "<Picture 2>", "kind": "picture", "role": "subject", "name": "Drummer"},
                {"label": "<Picture 3>", "kind": "picture", "role": "environment", "name": "Festival Stage"},
            ],
            profile="live_concert",
            actor_roles={"Singer": "Lead singer", "Drummer": "Drummer"},
            prop_bindings={"Singer": ("microphone",), "Drummer": ("drum kit",)},
        )

        self.assertIn("main festival stage", contract)
        self.assertIn("no catwalk, podium, or satellite platform", contract)
        self.assertIn("Singer remains bound to microphone", contract)
        self.assertIn("Drummer remains bound to drum kit", contract)
        self.assertIn("Singer remains bound to microphone", contract)

    def test_unknown_profile_does_not_enable_live_concert_rules(self):
        contract = render_reference_contract(
            [{"label": "<Picture 1>", "kind": "picture", "role": "subject", "name": "Singer"}],
            profile="crowded_tavern",
            actor_roles={"Singer": "Lead singer"},
            prop_bindings={"Singer": ("microphone",)},
        )

        self.assertNotIn("catwalk", contract.lower())
        self.assertNotIn("microphone", contract.lower())

    def test_live_concert_profile_infers_standard_role_props_when_not_explicitly_bound(self):
        contract = render_reference_contract(
            [{"label": "<Picture 1>", "kind": "picture", "role": "subject", "name": "Lead"}],
            profile="live_concert",
            actor_roles={"Lead": "Lead singer and frontman"},
        )

        self.assertIn("Lead remains bound to microphone", contract)


if __name__ == "__main__":
    unittest.main()
