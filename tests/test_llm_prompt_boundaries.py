from __future__ import annotations

import ast
from pathlib import Path
import unittest

from feverslop.prompting.general_signatures import build_general_signature_bundle
from feverslop.prompting.guide_loader import load_markdown_guide


class LlmPromptBoundaryTests(unittest.TestCase):
    def test_production_direct_completion_calls_only_live_in_transport(self):
        root = Path(__file__).parents[1] / "src"
        allowed = {
            root / "feverslop" / "prompting" / "dspy_runtime.py",
            root / "feverslop" / "adapters" / "llm_client.py",
        }
        violations = []
        for path in root.rglob("*.py"):
            if path in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "complete_prompt",
                    "complete_prompt_with_images",
                    "create",
                }:
                    is_transport_call = node.func.attr in {"complete_prompt", "complete_prompt_with_images"}
                    is_raw_openai_call = ast.unparse(node.func).endswith(".chat.completions.create")
                    if is_transport_call or is_raw_openai_call:
                        violations.append(f"{path}:{node.lineno}")
        self.assertEqual([], violations)

    def test_remaining_contracts_have_typed_signatures_and_guides(self):
        bundle = build_general_signature_bundle()
        self.assertEqual({"song_brief", "lyric_alignment", "zimage_prompt", "i2v_prompt", "storyboard_transform"}, set(bundle))
        for guide in ("song-brief", "lyric-alignment", "storyboard-transform"):
            self.assertTrue(load_markdown_guide(guide).strip())
