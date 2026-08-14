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
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
                    if len(node.args) >= 2:
                        target = node.args[0]
                        name = node.args[1]
                        if isinstance(name, ast.Constant):
                            value = name.value
                            if isinstance(value, str) and value.startswith("complete"):
                                violations.append(f"{path}:{node.lineno}")
                        elif isinstance(target, ast.Name) and target.id in {"llm", "client", "transport"}:
                            violations.append(f"{path}:{node.lineno}")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "complete_prompt",
                    "complete_prompt_with_images",
                }:
                    violations.append(f"{path}:{node.lineno}")
        self.assertEqual([], violations)

    def test_remaining_contracts_have_typed_signatures_and_guides(self):
        bundle = build_general_signature_bundle()
        self.assertEqual({"song_brief", "lyric_alignment", "zimage_prompt", "i2v_prompt", "storyboard_transform"}, set(bundle))
        guides = {
            "song_brief": "song-brief",
            "lyric_alignment": "lyric-alignment",
            "zimage_prompt": "music-video-t2i",
            "i2v_prompt": "music-video-i2v",
            "storyboard_transform": "storyboard-transform",
        }
        for contract, guide in guides.items():
            self.assertTrue(load_markdown_guide(guide).strip(), contract)
