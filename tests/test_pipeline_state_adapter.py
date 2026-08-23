import unittest

from feverslop.adapters.pipeline_state_store import PipelineStateStore


class PipelineStateAdapterTests(unittest.TestCase):
    def test_pipeline_state_store_is_available_from_adapters(self):
        self.assertTrue(callable(PipelineStateStore))


if __name__ == "__main__":
    unittest.main()
