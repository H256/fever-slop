import unittest

from feverslop.adapters.media_store import MediaStore
from feverslop.studio.media_store import MediaStore as LegacyMediaStore


class MediaPersistenceAdapterTests(unittest.TestCase):
    def test_media_store_has_canonical_adapter_owner(self):
        self.assertIs(LegacyMediaStore, MediaStore)


if __name__ == "__main__":
    unittest.main()
