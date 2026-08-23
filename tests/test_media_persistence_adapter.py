import unittest

from feverslop.adapters.media_store import MediaStore


class MediaPersistenceAdapterTests(unittest.TestCase):
    def test_media_store_is_available_from_adapters(self):
        self.assertTrue(callable(MediaStore))


if __name__ == "__main__":
    unittest.main()
