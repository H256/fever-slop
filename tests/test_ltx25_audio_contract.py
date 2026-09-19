import json
import unittest
from pathlib import Path

from feverslop.composition.config_loader import runner_root
from feverslop.domain.ltx25_audio_contract import (
    LTX25AudioContractError,
    LTX25AudioPolicy,
    load_ltx25_audio_policy,
    validate_ltx25_audio_workflow,
)


def _load_workflow(mode: str, quality: str) -> dict:
    path = runner_root() / "workflows" / "video" / "ltx_25" / mode / f"{mode}_{quality}.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


class LoadPolicyTests(unittest.TestCase):
    def test_missing_sidecar_defaults_to_not_applicable(self):
        policy = load_ltx25_audio_policy("/nonexistent/t2v_draft.json")
        self.assertEqual("not_applicable", policy.audio_policy)
        self.assertFalse(policy.preserve_audio_latent)

    def test_real_sidecar_is_read(self):
        path = (
            runner_root()
            / "workflows" / "video" / "ltx_25" / "t2v" / "t2v_draft.json"
        )
        policy = load_ltx25_audio_policy(path)
        self.assertEqual("native_audio_when_declared", policy.audio_policy)
        self.assertEqual("absolute_frame_windows", policy.timing_policy)
        self.assertTrue(policy.preserve_audio_latent)

    def test_all_real_sidecars_declare_native_audio(self):
        for mode in ("t2v", "i2v", "msr", "ingredients"):
            for quality in ("draft", "standard", "final"):
                path = (
                    runner_root()
                    / "workflows" / "video" / "ltx_25" / mode / f"{mode}_{quality}.json"
                )
                policy = load_ltx25_audio_policy(path)
                self.assertEqual("native_audio_when_declared", policy.audio_policy)
                self.assertTrue(policy.preserve_audio_latent)


class ValidateWorkflowTests(unittest.TestCase):
    def test_all_real_workflows_satisfy_their_declared_policy(self):
        for mode in ("t2v", "i2v", "msr", "ingredients"):
            for quality in ("draft", "standard", "final"):
                path = (
                    runner_root()
                    / "workflows" / "video" / "ltx_25" / mode / f"{mode}_{quality}.json"
                )
                policy = load_ltx25_audio_policy(path)
                validate_ltx25_audio_workflow(_load_workflow(mode, quality), policy)

    def test_not_applicable_policy_is_always_satisfiable(self):
        validate_ltx25_audio_workflow({}, LTX25AudioPolicy())

    def test_unsupported_audio_policy_is_rejected(self):
        with self.assertRaises(LTX25AudioContractError) as ctx:
            validate_ltx25_audio_workflow(
                _load_workflow("t2v", "draft"),
                LTX25AudioPolicy(audio_policy="weird"),
            )
        self.assertEqual("ltx25_audio_unsupported_policy", ctx.exception.code)

    def test_unsupported_timing_policy_is_rejected(self):
        with self.assertRaises(LTX25AudioContractError) as ctx:
            validate_ltx25_audio_workflow(
                _load_workflow("t2v", "draft"),
                LTX25AudioPolicy(audio_policy="native_audio_when_declared", timing_policy="weird"),
            )
        self.assertEqual("ltx25_timing_unsupported_policy", ctx.exception.code)

    def test_missing_audio_source_is_rejected(self):
        workflow = {
            k: v
            for k, v in _load_workflow("t2v", "draft").items()
            if v.get("class_type") not in {"LoadAudio", "TrimAudioDuration", "LTXVAudioVAEEncode"}
        }
        with self.assertRaises(LTX25AudioContractError) as ctx:
            validate_ltx25_audio_workflow(
                workflow,
                LTX25AudioPolicy(audio_policy="native_audio_when_declared"),
            )
        self.assertEqual("ltx25_audio_missing_source", ctx.exception.code)

    def test_broken_two_pass_chain_is_rejected(self):
        workflow = dict(_load_workflow("t2v", "draft"))
        decode_id = next(k for k, v in workflow.items() if v.get("class_type") == "LTXVAudioVAEDecode")
        separate_id = workflow[decode_id]["inputs"]["samples"][0]
        workflow[separate_id] = {"class_type": "VRAMCleanup", "inputs": {"anything": [separate_id, 0]}}
        with self.assertRaises(LTX25AudioContractError) as ctx:
            validate_ltx25_audio_workflow(
                workflow,
                LTX25AudioPolicy(audio_policy="native_audio_when_declared", preserve_audio_latent=True),
            )
        self.assertEqual("ltx25_audio_latent_not_preserved", ctx.exception.code)

    def test_missing_output_audio_is_rejected(self):
        workflow = {
            k: v
            for k, v in _load_workflow("t2v", "draft").items()
            if v.get("class_type") not in {"VHS_VideoCombine", "CreateVideo"}
        }
        with self.assertRaises(LTX25AudioContractError) as ctx:
            validate_ltx25_audio_workflow(
                workflow,
                LTX25AudioPolicy(audio_policy="native_audio_when_declared"),
            )
        self.assertEqual("ltx25_audio_missing_output", ctx.exception.code)


if __name__ == "__main__":
    unittest.main()
