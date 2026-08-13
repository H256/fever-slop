import unittest


class AuditMovieConfigPathTests(unittest.TestCase):
    def test_movie_adapters_use_explicit_app_config_path(self):
        from feverslop.composition.movie_pipeline import _movie_app_config_path

        self.assertEqual("custom/app.json", _movie_app_config_path({"app_config_path": "custom/app.json"}))
        self.assertEqual("app_config.json", _movie_app_config_path({}))


if __name__ == "__main__":
    unittest.main()
