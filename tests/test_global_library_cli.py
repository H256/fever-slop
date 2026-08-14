import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from feverslop.tools.global_library_cli import main


class GlobalLibraryCliTests(unittest.TestCase):
    def test_create_list_and_show_json(self):
        with tempfile.TemporaryDirectory() as temp:
            args = ["--library-root", temp, "create", "--kind", "prop", "--id", "lamp", "--name", "Lamp"]
            self.assertEqual(0, main(args))
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(["--library-root", temp, "list", "--json"]))
            self.assertEqual("lamp", json.loads(output.getvalue())[0]["id"])

    def test_invalid_input_returns_nonzero_and_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            error = StringIO()
            with redirect_stderr(error):
                code = main(["--library-root", temp, "show", "--kind", "prop", "--id", "missing"])
            self.assertNotEqual(0, code)
            self.assertIn("create or import", error.getvalue())


if __name__ == "__main__":
    unittest.main()
