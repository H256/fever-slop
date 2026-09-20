import json
import unittest
from pathlib import Path

from feverslop.domain.h3_two_pass import (
    H3TwoPassSchemaError,
    H3TwoPassSpec,
    validate_h3_two_pass_topology,
)


class H3TwoPassTests(unittest.TestCase):
    def make_spec(self, **overrides):
        values = {
            "model_assets": ["minimax_h3", "vae"],
            "pass1_sampler": "euler",
            "pass1_scheduler": "normal",
            "pass1_steps": 20,
            "pass1_denoise": 1.0,
            "pass2_sampler": "euler_cfg1a",
            "pass2_scheduler": "sgm_uniform",
            "pass2_steps": 8,
            "pass2_denoise": 0.35,
            "preserve_audio_latent": True,
            "required_anchors": ["#PROMPT", "#FRAMECOUNT", "#PASS1", "#PASS2", "#AUDIO_LATENT"],
        }
        values.update(overrides)
        return H3TwoPassSpec.create(**values)

    def test_normalizes_and_serializes_two_pass_contract(self):
        spec = self.make_spec(pass1_sampler=" EULER ", model_assets=["vae", "minimax_h3", "vae"])

        self.assertEqual(("minimax_h3", "vae"), spec.model_assets)
        self.assertEqual("euler", spec.pass1_sampler)
        self.assertTrue(spec.preserve_audio_latent)
        self.assertEqual(spec, H3TwoPassSpec.from_dict(spec.to_dict()))

    def test_rejects_invalid_pass_parameters_and_three_pass_shape(self):
        for overrides in (
            {"pass1_steps": 0},
            {"pass2_denoise": 1.1},
            {"required_anchors": ["#PROMPT", "#PASS3"]},
            {"preserve_audio_latent": "yes"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(H3TwoPassSchemaError):
                    self.make_spec(**overrides)

    def test_validates_required_workflow_anchors(self):
        spec = self.make_spec()

        spec.validate_workflow_anchors({"#PROMPT", "#FRAMECOUNT", "#PASS1", "#PASS2", "#AUDIO_LATENT"})
        with self.assertRaises(H3TwoPassSchemaError):
            spec.validate_workflow_anchors({"#PROMPT", "#FRAMECOUNT", "#PASS1"})

    def test_default_spec_is_available_for_every_quality_profile(self):
        from feverslop.domain.h3_two_pass import default_h3_two_pass_spec

        for quality in ("draft", "standard", "final"):
            spec = default_h3_two_pass_spec(quality, audio=True)
            self.assertGreater(spec.pass1_steps, 0)
            self.assertTrue(spec.preserve_audio_latent)
            self.assertIn("#PASS2", spec.required_anchors)

    def test_topology_requires_separation_upscale_recombine_and_refinement(self):
        spec = self.make_spec(required_anchors=["#PASS1", "#PASS2"], preserve_audio_latent=False)
        workflow = {
            "1": {"class_type": "MiniMaxH3AVLatentSeparateT8", "inputs": {}, "_meta": {"title": "#SEPARATE_AV"}},
            "2": {"class_type": "VRGDG_MiniMaxH3LearnedLatentUpscale", "inputs": {}, "_meta": {"title": "#LATENT_UPSCALE"}},
            "3": {"class_type": "VRGDG_MiniMaxH3ReplaceUpscaledVideoLatent", "inputs": {}, "_meta": {"title": "#RECOMBINE_AV"}},
            "4": {"class_type": "SamplerCustomAdvanced", "inputs": {}, "_meta": {"title": "#PASS1"}},
            "5": {"class_type": "SamplerCustomAdvanced", "inputs": {}, "_meta": {"title": "#PASS2"}},
        }
        validate_h3_two_pass_topology(workflow, spec)


class H3QualityRegistryTests(unittest.TestCase):
    def test_registry_covers_three_profiles_with_monotone_budgets(self):
        from feverslop.domain.h3_two_pass import h3_quality_budgets

        budgets = h3_quality_budgets()
        self.assertEqual({"draft", "standard", "final"}, set(budgets))
        tiers = ["draft", "standard", "final"]
        for field in ("pass1_steps", "pass2_steps"):
            values = [budgets[t][field] for t in tiers]
            self.assertEqual(sorted(values), values)
        for tier in tiers:
            self.assertEqual(1.0, budgets[tier]["pass1_denoise"])
            self.assertGreater(budgets[tier]["pass2_max_megapixels"], budgets[tier]["pass1_megapixels"])

    def test_registry_matches_calibrated_default_spec(self):
        from feverslop.domain.h3_two_pass import default_h3_two_pass_spec, h3_quality_budgets

        for quality, budget in h3_quality_budgets().items():
            spec = default_h3_two_pass_spec(quality)
            self.assertEqual(int(budget["pass1_steps"]), spec.pass1_steps)
            self.assertEqual(float(budget["pass1_denoise"]), spec.pass1_denoise)
            self.assertEqual(int(budget["pass2_steps"]), spec.pass2_steps)
            self.assertEqual(float(budget["pass2_denoise"]), spec.pass2_denoise)

    def test_quality_extraction_from_render_profile(self):
        from feverslop.domain.h3_two_pass import quality_from_render_profile

        self.assertEqual("draft", quality_from_render_profile("ltx25-r2v-draft"))
        self.assertEqual("standard", quality_from_render_profile("ltx25-r2v-standard"))
        self.assertEqual("final", quality_from_render_profile("ltx25-r2v-final"))
        self.assertEqual("draft", quality_from_render_profile("ltx25-r2v-unknown"))
        self.assertEqual("draft", quality_from_render_profile(""))
        self.assertEqual("draft", quality_from_render_profile(None))


class H3TwoPassBudgetWiringTests(unittest.TestCase):
    """The selected quality must actually change the two-pass sampling budget.

    Exercises the production helper against the real two-pass template so the
    profile name has a real effect (the root cause: the template carried one
    static budget matching no single profile).
    """

    TEMPLATE = Path(__file__).resolve().parents[1] / "workflows" / "video" / "minimax_h3" / "t2v_two_pass.json"

    def _budget(self, patcher) -> tuple:
        _, n1 = patcher.find_node_by_meta_title("#PASS1_SCHEDULER")
        _, n2 = patcher.find_node_by_meta_title("#PASS2_SCHEDULER")
        return (
            n1["inputs"]["steps"], n1["inputs"]["denoise"],
            n2["inputs"]["steps"], n2["inputs"]["denoise"],
        )

    def _patched_budget(self, quality: str) -> tuple:
        from feverslop.adapters.comfyui_minimax_h3_video_backend import ComfyUIMiniMaxH3VideoRenderBackend
        from feverslop.adapters.workflow_patcher import WorkflowPatcher
        from feverslop.domain.h3_two_pass import default_h3_two_pass_spec

        patcher = WorkflowPatcher(json.loads(self.TEMPLATE.read_text()))
        backend = ComfyUIMiniMaxH3VideoRenderBackend.__new__(ComfyUIMiniMaxH3VideoRenderBackend)
        backend._patch_two_pass_budget(patcher, default_h3_two_pass_spec(quality))
        return self._budget(patcher)

    def test_final_quality_patches_real_template_to_final_budget(self):
        # final = pass1 28 steps / denoise 1.0, pass2 12 steps / denoise 0.30
        self.assertEqual((28, 1.0, 12, 0.30), self._patched_budget("final"))

    def test_draft_quality_patches_real_template_to_draft_budget(self):
        # draft = pass1 12 steps / denoise 1.0, pass2 4 steps / denoise 0.55
        self.assertEqual((12, 1.0, 4, 0.55), self._patched_budget("draft"))

    def test_three_qualities_produce_distinct_budgets(self):
        self.assertNotEqual(self._patched_budget("draft"), self._patched_budget("standard"))
        self.assertNotEqual(self._patched_budget("standard"), self._patched_budget("final"))
        self.assertNotEqual(self._patched_budget("draft"), self._patched_budget("final"))

    def test_single_pass_template_is_no_op(self):
        from feverslop.adapters.comfyui_minimax_h3_video_backend import ComfyUIMiniMaxH3VideoRenderBackend
        from feverslop.adapters.workflow_patcher import WorkflowPatcher
        from feverslop.domain.h3_two_pass import default_h3_two_pass_spec

        # a single-pass template lacks the #PASS1_SCHEDULER anchor
        single_pass = {
            "1": {"class_type": "KSampler", "inputs": {"steps": 20, "denoise": 1.0}},
        }
        patcher = WorkflowPatcher(single_pass)
        # grab a real backend instance just for the helper (no client needed)
        backend = ComfyUIMiniMaxH3VideoRenderBackend.__new__(ComfyUIMiniMaxH3VideoRenderBackend)
        backend._patch_two_pass_budget(patcher, default_h3_two_pass_spec("final"))
        # unchanged
        self.assertEqual(20, single_pass["1"]["inputs"]["steps"])
        self.assertEqual(1.0, single_pass["1"]["inputs"]["denoise"])

    def test_single_pass_turbo_template_keeps_template_steps(self):
        # Regression: the full-mix turbo template is single-pass but carried a
        # #PASS1_SCHEDULER anchor, so the quality budget overwrote its 8 steps.
        from feverslop.adapters.comfyui_minimax_h3_video_backend import ComfyUIMiniMaxH3VideoRenderBackend
        from feverslop.adapters.workflow_patcher import WorkflowPatcher
        from feverslop.domain.h3_two_pass import default_h3_two_pass_spec

        turbo = (
            Path(__file__).resolve().parents[1]
            / "workflows" / "video" / "minimax_h3"
            / "r2v_audio_fullmix_guide_1-pass_turbo.json"
        )
        backend = ComfyUIMiniMaxH3VideoRenderBackend.__new__(ComfyUIMiniMaxH3VideoRenderBackend)
        for quality in ("draft", "standard", "final"):
            with self.subTest(quality=quality):
                patcher = WorkflowPatcher(json.loads(turbo.read_text(encoding="utf-8")))
                backend._patch_two_pass_budget(patcher, default_h3_two_pass_spec(quality))
                schedulers = [
                    node for node in patcher.get().values()
                    if node.get("class_type") == "BasicScheduler"
                ]
                self.assertEqual(1, len(schedulers))
                self.assertEqual(8, schedulers[0]["inputs"]["steps"])
                self.assertEqual(1, schedulers[0]["inputs"]["denoise"])


if __name__ == "__main__":
    unittest.main()
