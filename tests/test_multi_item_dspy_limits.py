import json
import unittest
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

from feverslop.prompting.general_modules import GeneralPromptModules
from feverslop.prompting.general_signatures import build_general_signature_bundle
from feverslop.prompting.msr_modules import MSRPromptModules
from feverslop.prompting.planning_payload import _EVIDENCE_FIELDS


def _evidence_segment() -> dict[str, Any]:
    return {
        "segment_id": "segment_001",
        "scene": "Mara walks through a rain-soaked alley",
        "start": 10.0,
        "end": 14.0,
        "duration": 4.0,
        "type": "vocals",
        "lyrics": "Mara sings into the rain",
        "performance_intervals": [
            {
                "phase": "singing",
                "word_timestamps": [
                    {"word": "Mara", "start": 10.0, "end": 10.4},
                    {"word": "sings", "start": 10.4, "end": 10.9},
                ],
                "vocal_sources": [
                    {"subject_id": "mara", "alignment": {"raw_text": "x" * 100000}},
                ],
                "vocal_events": [
                    {"word": "Mara", "start": 10.0, "end": 10.4},
                    {"word": "sings", "start": 10.4, "end": 10.9, "alignment": {"raw_text": "raw"}},
                ],
                "performance_conflicts": ["unresolved phase overlap"],
                "reason_codes": ["PHASE_MERGE"],
            }
        ],
        "word_timestamps": [
            {"word": "Mara", "start": 10.0, "end": 10.4},
            {"word": "sings", "start": 10.4, "end": 10.9},
        ],
        "performance_conflicts": ["unresolved phase overlap"],
        "reason_codes": ["PHASE_MERGE"],
        "ltx": {
            "prompt_relay": [
                {
                    "state": "singing",
                    "lyrics": "Mara sings into the rain",
                    "frame_start": 0,
                    "frame_end": 100,
                    "prompt": "relay prompt",
                }
            ]
        },
    }


def _collect_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                keys.add(key)
            keys.update(_collect_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_collect_keys(item))
    return keys


class MultiItemDspyLimitTests(unittest.TestCase):
    def _runtime(self, calls, result):
        class Predictor:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return result

        class Runtime:
            def make_lm(self, llm):
                return "lm"

            context = staticmethod(lambda **kwargs: nullcontext())
            predict = staticmethod(lambda signature: Predictor())

        return Runtime()

    def test_lyric_alignment_scales_limit_by_segment_count(self):
        calls = []

        class LLM:
            model = "fake-model"
            client = object()

        modules = GeneralPromptModules(
            LLM(),
            dspy_runtime=self._runtime(calls, {"result": {"segments": {}}}),
        )
        modules.lyric_alignment({"WHISPER_SEGMENTS": [{"key": str(i)} for i in range(13)]})

        self.assertEqual(15360, calls[0]["config"]["max_tokens"])

    def test_i2v_signature_transports_optional_performers_without_pydantic_validation(self):
        signature = build_general_signature_bundle()["i2v_prompt"]

        self.assertEqual(dict[str, Any], signature.fields["result"].annotation)

    def test_i2v_prompt_normalizes_single_text_field_and_missing_speaker_id(self):
        calls = []

        class LLM:
            model = "fake-model"
            client = object()

        modules = GeneralPromptModules(
            LLM(),
            dspy_runtime=self._runtime(calls, {
                "result": {
                    "generated_motion": "Mara sings into the rain.",
                    "vocal_performers": [
                        {"subject_id": "mara"},
                    ],
                },
            }),
        )

        result = modules.i2v_prompt({}, guide="test guide")

        self.assertEqual("Mara sings into the rain.", result.prompt)
        self.assertEqual(
            [{"subject_id": "mara", "speaker_id": "S1"}],
            [performer.model_dump() for performer in result.vocal_performers],
        )

    def test_msr_segments_scales_limit_by_relay_count(self):
        calls = []

        class LLM:
            model = "fake-model"
            client = object()

        modules = MSRPromptModules(
            LLM(),
            dspy_runtime=self._runtime(calls, {"result": {"references": [], "relays": []}}),
        )
        modules.segments({"relay_segments": [{"index": i} for i in range(4)]})

        self.assertEqual(10240, calls[0]["config"]["max_tokens"])

    def test_zimage_prompt_strips_replicated_evidence_keeps_semantic_signal(self):
        calls = []

        class LLM:
            model = "fake-model"
            client = object()

        modules = GeneralPromptModules(LLM(), dspy_runtime=self._runtime(calls, {"result": {"prompt": "T2I RESULT"}}))
        segment = _evidence_segment()
        original = deepcopy(segment)

        result = modules.zimage_prompt(
            {"segment": segment, "scene_concept": "A rain-soaked alley", "global_subject": "Mara"},
        )

        self.assertEqual("T2I RESULT", result.prompt)
        payload = calls[0]["payload"]
        self.assertFalse(_collect_keys(payload) & _EVIDENCE_FIELDS)
        self.assertNotIn("x" * 100000, json.dumps(payload))
        self.assertEqual("Mara sings into the rain", payload["segment"]["lyrics"])
        self.assertEqual("vocals", payload["segment"]["type"])
        self.assertEqual("A rain-soaked alley", payload["scene_concept"])
        self.assertEqual("singing", payload["segment"]["ltx"]["prompt_relay"][0]["state"])
        self.assertEqual(original, segment)

    def test_i2v_prompt_strips_replicated_evidence_keeps_semantic_signal(self):
        calls = []

        class LLM:
            model = "fake-model"
            client = object()

        modules = GeneralPromptModules(LLM(), dspy_runtime=self._runtime(calls, {"result": {"prompt": "I2V RESULT"}}))
        segment = _evidence_segment()
        original = deepcopy(segment)

        result = modules.i2v_prompt(
            {"segment": segment, "performance_policy": "singing policy"},
            guide="test guide",
        )

        self.assertEqual("I2V RESULT", result.prompt)
        self.assertEqual("test guide", calls[0]["guide"])
        payload = calls[0]["payload"]
        self.assertFalse(_collect_keys(payload) & _EVIDENCE_FIELDS)
        self.assertNotIn("x" * 100000, json.dumps(payload))
        self.assertEqual("Mara sings into the rain", payload["segment"]["lyrics"])
        self.assertEqual("vocals", payload["segment"]["type"])
        self.assertEqual("singing policy", payload["performance_policy"])
        self.assertEqual("singing", payload["segment"]["ltx"]["prompt_relay"][0]["state"])
        self.assertEqual(original, segment)

    def test_lyric_alignment_request_keeps_raw_word_timing(self):
        calls = []

        class LLM:
            model = "fake-model"
            client = object()

        modules = GeneralPromptModules(
            LLM(),
            dspy_runtime=self._runtime(calls, {"result": {"segments": {"segment1": "hello"}}}),
        )
        request = {
            "REFERENCE_LYRICS": "Mara sings into the rain",
            "WHISPER_SEGMENTS": [
                {
                    "key": "segment1",
                    "start": 10.0,
                    "end": 14.0,
                    "duration": 4.0,
                    "text": "Mara sings into the rain",
                    "word_timestamps": [
                        {"word": "Mara", "start": 10.0, "end": 10.4},
                        {"word": "sings", "start": 10.4, "end": 10.9},
                    ],
                    "alignment": {"raw_text": "Mara sings into the rain (raw whisper)"},
                }
            ],
        }

        modules.lyric_alignment(request)

        recorded = calls[0]["request"]["WHISPER_SEGMENTS"][0]
        self.assertEqual(request["WHISPER_SEGMENTS"][0]["word_timestamps"], recorded["word_timestamps"])
        self.assertEqual("Mara sings into the rain (raw whisper)", recorded["alignment"]["raw_text"])


if __name__ == "__main__":
    unittest.main()
