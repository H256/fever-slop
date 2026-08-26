import unittest

from feverslop.domain.continuation_dependencies import ContinuationDependencyGraph


class ContinuationDependencyGraphTests(unittest.TestCase):
    def test_successor_is_blocked_until_verified_predecessor(self):
        graph = ContinuationDependencyGraph.from_chains({"action": ["a-0001", "a-0002", "a-0003"]})
        self.assertEqual(("a-0001",), graph.ready())
        graph.mark_complete("a-0001", anchor_valid=True)
        self.assertEqual(("a-0002",), graph.ready())
        graph.mark_complete("a-0002", anchor_valid=True)
        self.assertEqual(("a-0003",), graph.ready())

    def test_invalidating_middle_segment_invalidates_only_suffix(self):
        graph = ContinuationDependencyGraph.from_chains({"action": ["a-0001", "a-0002", "a-0003"], "other": ["b-0001"]})
        graph.mark_complete("a-0001", anchor_valid=True)
        graph.mark_complete("a-0002", anchor_valid=True)
        graph.mark_complete("a-0003", anchor_valid=True)
        graph.mark_complete("b-0001", anchor_valid=True)
        self.assertEqual(("a-0002", "a-0003"), graph.invalidate_suffix("a-0002"))
        self.assertEqual(("a-0002",), graph.ready())


if __name__ == "__main__":
    unittest.main()
