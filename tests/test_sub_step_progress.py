import unittest
from unittest.mock import Mock

from feverslop.utils.sub_step_progress import SubStepProgress


class SubStepProgressTests(unittest.TestCase):
    def test_reports_at_interval_and_always_on_completion(self):
        reporter = Mock()
        progress = SubStepProgress(reporter, "Prompt Generation", 25, interval=10)

        for current in range(1, 26):
            progress.update(current)

        messages = [call.args[0] for call in reporter.message.call_args_list]
        self.assertEqual(3, len(messages))
        self.assertIn("Prompt Generation: 10/25", messages[0])
        self.assertIn("Prompt Generation: 20/25", messages[1])
        self.assertIn("Prompt Generation: 25/25", messages[2])
        self.assertRegex(messages[0], r"\[\d{2}:\d{2}\]")

    def test_verbose_reports_every_item(self):
        reporter = Mock()
        progress = SubStepProgress(reporter, "Scenes", 3, interval=10, verbose=True)
        for current in range(1, 4):
            progress.update(current)
        self.assertEqual(3, reporter.message.call_count)

    def test_quiet_suppresses_all_messages(self):
        reporter = Mock()
        progress = SubStepProgress(reporter, "Scenes", 3, quiet=True)
        progress.update(1, force=True)
        progress.update(3, force=True)
        reporter.message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
