import unittest

from feverslop.adapters.pipeline_state_store import PipelineStateStore
from feverslop.studio.pipeline_state_store import PipelineStateStore as LegacyPipelineStateStore


class PipelineStateAdapterTests(unittest.TestCase):
    def test_pipeline_state_store_has_canonical_adapter_owner(self):
        self.assertIs(LegacyPipelineStateStore, PipelineStateStore)


if __name__ == "__main__":
    unittest.main()
