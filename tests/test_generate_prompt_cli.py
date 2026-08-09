import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest


from tools import generate_prompt


class FakeConfig:
    class LLM:
        base_url = "http://config.example/v1"
        model = "config-model"
        api_key = "config-secret"
        temperature = 0.25
        max_tokens = 123
        request_timeout_seconds = 45.0

    llm = LLM()


class FakeResult:
    rendered_prompt = "integrated_multimodal_description: generated\noverall_soundscape: quiet"


class GeneratePromptCliTests(unittest.TestCase):
    def test_parser_requires_model_type_and_description(self):
        parser = generate_prompt.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args([])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model-type", "minimax-h3-t2v"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--description", "A dancer."])

    def test_parser_accepts_minimax_options_and_repeated_reference_json(self):
        args = generate_prompt.build_arg_parser().parse_args([
            "--model-type", "minimax-h3-r2v",
            "--description", "A dancer crosses a neon street.",
            "--reference", '{"kind":"picture","source":"first.png","role":"subject"}',
            "--reference", '{"kind":"video","source":"motion.mp4","role":"motion"}',
            "--duration", "6.5",
            "--notes", "Keep the camera low.",
            "--music-intent", "reference",
            "--no-strict-fidelity",
        ])

        self.assertEqual("minimax-h3-r2v", args.model_type)
        self.assertEqual("A dancer crosses a neon street.", args.description)
        self.assertEqual(2, len(args.reference))
        self.assertEqual(6.5, args.duration)
        self.assertEqual("Keep the camera low.", args.notes)
        self.assertEqual("reference", args.music_intent)
        self.assertFalse(args.strict_fidelity)

    def test_reference_json_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.json"
            reference_path.write_text(
                json.dumps({"kind": "picture", "source": "style.png", "role": "style"}),
                encoding="utf-8",
            )

            args = generate_prompt.build_arg_parser().parse_args([
                "--model-type", "minimax-h3-t2v",
                "--description", "A city at night.",
                "--reference", str(reference_path),
            ])

            self.assertEqual(str(reference_path), args.reference[0])
            self.assertEqual(
                [{"kind": "picture", "source": "style.png", "role": "style"}],
                generate_prompt.load_references(args.reference),
            )

    def test_main_renders_prompt_with_injected_factories_and_no_network(self):
        calls = {}

        def client_factory(**kwargs):
            calls["client"] = kwargs
            return object()

        def generator_factory(client):
            calls["generator_client"] = client
            return object()

        class RecordingService:
            def __init__(self, generator):
                calls["service_generator"] = generator

            def generate(self, model_type, description, **kwargs):
                calls["request"] = (model_type, description, kwargs)
                return FakeResult()

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = generate_prompt.main(
                [
                    "--model-type", "minimax-h3-t2v",
                    "--description", "A dancer.",
                    "--reference", '{"kind":"picture","source":"style.png","role":"style"}',
                    "--duration", "5",
                    "--api-key", "injected-secret",
                ],
                config_loader=lambda path: FakeConfig(),
                client_factory=client_factory,
                generator_factory=generator_factory,
                service_factory=RecordingService,
            )

        self.assertEqual(0, result)
        self.assertEqual(FakeResult.rendered_prompt + "\n", stdout.getvalue())
        self.assertEqual("", stderr.getvalue())
        self.assertEqual("injected-secret", calls["client"]["api_key"])
        self.assertEqual("http://config.example/v1", calls["client"]["base_url"])
        self.assertEqual(("minimax-h3-t2v", "A dancer.", {
            "references": [{"kind": "picture", "source": "style.png", "role": "style"}],
            "duration_seconds": 5.0,
            "notes": None,
            "music_intent": None,
            "strict_fidelity": True,
        }), calls["request"])

    def test_invalid_model_type_returns_concise_error_without_secret(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = generate_prompt.main([
                "--model-type", "invalid-model",
                "--description", "A dancer.",
                "--api-key", "never-print-this-secret",
            ])

        self.assertNotEqual(0, result)
        self.assertEqual("", stdout.getvalue())
        self.assertIn("invalid-model", stderr.getvalue())
        self.assertNotIn("never-print-this-secret", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()