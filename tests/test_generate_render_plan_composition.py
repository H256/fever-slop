import unittest
from types import SimpleNamespace


class GenerateRenderPlanCompositionTests(unittest.TestCase):
    def test_scene_generation_seed_minus_one_randomizes_concrete_seed(self):
        from feverslop.composition.generate_render_plan import _build_scene_generator

        generator = _build_scene_generator(
            SimpleNamespace(
                min_duration=2.0,
                max_duration=10.0,
                bias=0.7,
                duration_preset="impact_weighted",
                seed=-1,
            )
        )

        self.assertIsInstance(generator.seed, int)
        self.assertNotEqual(-1, generator.seed)


if __name__ == "__main__":
    unittest.main()
