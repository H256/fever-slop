import unittest
from contextlib import nullcontext

from feverslop.prompting.dspy_h3_generator_core import VideoPromptGenerator
from feverslop.prompting.dspy_h3_models import (
    BaseVideoPrompt,
    H3CreativePlan,
    H3CreativeShot,
    MusicIntent,
    ResolvedPromptPlan,
    ResolvedReference,
)
from feverslop.prompting.dspy_runtime import DspyRuntime, H3SignatureBundle


class DspyRuntimeTests(unittest.TestCase):
    def test_make_lm_uses_injected_factory_and_openai_compatible_llm_settings(self):
        calls = []

        class Client:
            base_url = object()
            api_key = "local-key"

        class LLM:
            client = Client()
            model = "gemma4-26b-a4b"
            max_tokens = 2048
            request_timeout_seconds = 42.0
            dspy_temperature = 0.2
            dspy_cache = True

        runtime = DspyRuntime(
            signatures=H3SignatureBundle(object, object, object, object),
            lm_factory=lambda *args, **kwargs: calls.append((args, kwargs)) or "lm",
            predict_factory=lambda signature: signature,
            context_factory=lambda **kwargs: nullcontext(kwargs),
        )

        self.assertEqual("lm", runtime.make_lm(LLM()))
        self.assertEqual(("openai/gemma4-26b-a4b",), calls[0][0])
        self.assertEqual(
            {
                "api_base": str(Client.base_url),
                "api_key": "local-key",
                "temperature": 0.2,
                "max_tokens": 2048,
                "timeout": 42.0,
                "cache": True,
                "num_retries": 3,
            },
            calls[0][1],
        )

    def test_make_lm_injects_hardened_openai_client(self):
        from openai import OpenAI

        calls = []
        hardened_client = OpenAI(base_url="http://localhost:9/v1", api_key="k")

        class LLM:
            client = hardened_client
            model = "gemma4-26b-a4b"
            max_tokens = 2048
            max_retries = 2
            dspy_cache = True

        runtime = DspyRuntime(
            signatures=H3SignatureBundle(object, object, object, object),
            lm_factory=lambda *args, **kwargs: calls.append((args, kwargs)) or "lm",
            predict_factory=lambda signature: signature,
            context_factory=lambda **kwargs: nullcontext(kwargs),
        )

        self.assertEqual("lm", runtime.make_lm(LLM()))
        self.assertIs(hardened_client, calls[0][1]["client"])
        self.assertEqual(2, calls[0][1]["num_retries"])
        self.assertFalse(calls[0][1]["cache"])

    def test_generator_accepts_fake_runtime_and_loads_h3_guides_without_live_endpoint(self):
        class FakePredict:
            def __init__(self, signature):
                self.signature = signature
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                if self.signature == "Plan":
                    return type("Prediction", (), {
                        "plan": H3CreativePlan(
                            creative_intent="intent",
                            shots=[H3CreativeShot(description="A dancer moves.")],
                            overall_soundscape="quiet",
                            music_intent=MusicIntent.NONE,
                        ),
                    })()
                return type("Prediction", (), {
                    "result": BaseVideoPrompt(
                        integrated_multimodal_description="generated",
                        overall_soundscape="quiet",
                    ),
                })()

        predictions = []
        runtime = DspyRuntime(
            signatures=H3SignatureBundle("Analyze", "Plan", "Base", "Reference"),
            lm_factory=lambda *args, **kwargs: "lm",
            predict_factory=lambda signature: predictions.append(FakePredict(signature)) or predictions[-1],
            context_factory=lambda **kwargs: nullcontext(),
        )

        class LLM:
            client = None
            model = "fake"
            max_tokens = 128

        generator = VideoPromptGenerator(
            base_guide_path="minimax-h3-base.md",
            reference_guide_path="minimax-h3-references.md",
            llm=LLM(),
            dspy_runtime=runtime,
        )

        result = generator({
            "mode": "t2v",
            "user_prompt": "A dancer moves.",
            "references": [{"kind": "picture", "source": "actor.png", "description": "actor"}],
            "relay_segments": [{"shot": 1, "start_seconds": 0.0, "end_seconds": 1.0}],
        })

        self.assertEqual("generated", result.prompt.integrated_multimodal_description)
        plan_call = next(predict.calls[0] for predict in predictions if predict.signature == "Plan")
        base_call = next(predict.calls[0] for predict in predictions if predict.signature == "Base")
        self.assertIsInstance(plan_call["references"][0], ResolvedReference)
        self.assertEqual([{"shot": 1, "start_seconds": 0.0, "end_seconds": 1.0}], plan_call["relay_segments"])
        self.assertIsInstance(base_call["plan"], ResolvedPromptPlan)
        self.assertIsInstance(base_call["references"][0], ResolvedReference)
        self.assertNotIn("plan_json", base_call)
        self.assertNotIn("references_json", base_call)
        self.assertNotIn("relay_segments_json", base_call)
        self.assertIn("integrated_multimodal_description", base_call["guide"])

    def test_dspy_lm_does_not_follow_redirects(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        import dspy
        import httpx
        from openai import OpenAI

        class Handler(BaseHTTPRequestHandler):
            seen_paths: list[str] = []

            def do_POST(self):
                Handler.seen_paths.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
                if self.path == "/v1/chat/completions":
                    self.send_response(307)
                    self.send_header("Location", "/target")
                    self.end_headers()
                    return
                body = json.dumps({
                    "id": "cmpl-redirect",
                    "object": "chat.completion",
                    "model": "fake-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        hardened_client = OpenAI(
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            api_key="k",
            http_client=httpx.Client(follow_redirects=False),
            max_retries=0,
        )

        class LLM:
            client = hardened_client
            model = "fake-model"
            max_tokens = 64
            # Small budget keeps the litellm retry backoff short in tests.
            max_retries = 1
            dspy_cache = False

        try:
            lm = DspyRuntime.create(dspy).make_lm(LLM())
            with self.assertRaises(dspy.LMProviderError) as caught:
                lm("hi")
        finally:
            server.shutdown()

        self.assertIn("307", str(caught.exception))
        self.assertNotIn("/target", Handler.seen_paths)


if __name__ == "__main__":
    unittest.main()
