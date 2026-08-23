import unittest


class TestMoviePipelineModes(unittest.TestCase):
    def test_minimax_h3_modes_do_not_use_ltx_msr_enrichment(self):
        from feverslop.composition.movie_pipeline import (
            _movie_uses_msr_reference_enrichment,
        )

        self.assertFalse(_movie_uses_msr_reference_enrichment("minimax-h3-r2v"))
        self.assertFalse(_movie_uses_msr_reference_enrichment("minimax-h3-t2v"))
        self.assertFalse(_movie_uses_msr_reference_enrichment("minimax-h3-i2v"))
        self.assertTrue(_movie_uses_msr_reference_enrichment("msr"))
        self.assertTrue(_movie_uses_msr_reference_enrichment("msr-i2v-startframe"))

    def test_default_movie_stage_titles_include_render_and_completion(self):
        from feverslop.composition.movie_pipeline import _movie_stage_titles

        minimax_titles = _movie_stage_titles({"movie_video_workflow": "minimax-h3-r2v"})
        self.assertIn("Movie MiniMax H3 render", minimax_titles)
        self.assertIn("Movie complete", minimax_titles)

        msr_titles = _movie_stage_titles({"movie_video_workflow": "msr"})
        self.assertIn("Movie MSR render", msr_titles)
        self.assertIn("Movie complete", msr_titles)


if __name__ == "__main__":
    unittest.main()
