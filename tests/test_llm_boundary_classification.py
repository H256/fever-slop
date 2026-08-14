"""Static guards for production LLM transport boundaries."""

import ast
from pathlib import Path
import unittest


PRODUCTION_ROOT = Path("src/feverslop")
APPROVED_DIRECT_LLM_CALLS = {
    "src/feverslop/adapters/gemma4_startframe_validator.py": {
        "requests.post",
    },
}


class LLMBoundaryClassificationTests(unittest.TestCase):
    def test_direct_llm_transport_is_explicitly_classified(self):
        offenders = []
        for path in PRODUCTION_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            allowed = APPROVED_DIRECT_LLM_CALLS.get(path.as_posix(), set())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if not isinstance(node.func.value, ast.Name):
                    continue
                call = f"{node.func.value.id}.{node.func.attr}"
                if call == "requests.post" and call not in allowed:
                    offenders.append(f"{path.as_posix()}:{node.lineno}:{call}")
        self.assertEqual([], offenders)

    def test_all_openai_compatible_client_construction_passes_dspy_temperature(self):
        missing = []
        for path in PRODUCTION_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id != "OpenAICompatibleLLMClient":
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                value = keywords.get("dspy_temperature")
                if not isinstance(value, ast.Attribute) or value.attr != "dspy_temperature":
                    missing.append(f"{path}:{node.lineno}")
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
