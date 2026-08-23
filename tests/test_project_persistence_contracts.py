import unittest

from feverslop.application.project_requests import sanitize_audio_filename


class ProjectPersistenceContractTests(unittest.TestCase):
    def test_persistence_contracts_have_canonical_application_owner(self):
        self.assertEqual("bad_name.mp3", sanitize_audio_filename(r"../bad name.mp3"))


if __name__ == "__main__":
    unittest.main()
