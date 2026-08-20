import unittest

from feverslop.adapters.movie_workflow import MovieWorkflowPatcher


class MovieWorkflowAudioStripTests(unittest.TestCase):
    def _build_workflow(self, consumer_inputs):
        return {
            "1": {
                "class_type": "LoadAudio",
                "inputs": {"audio": "song.mp3"},
                "_meta": {"title": "#LOAD_AUDIO"},
            },
            "2": {
                "class_type": "TrimAudio",
                "inputs": {"audio": ["1", 0]},
                "_meta": {"title": "#TRIM_AUDIO"},
            },
            "40": {"class_type": "LoadImage", "inputs": {}, "_meta": {"title": "#SOURCE"}},
            "5": {"class_type": "KSampler", "inputs": consumer_inputs},
        }

    def _strip(self, consumer_inputs):
        workflow = self._build_workflow(consumer_inputs)
        return MovieWorkflowPatcher().strip_audio_inputs(workflow)

    def test_flat_link_to_removed_node_deletes_input(self):
        result = self._strip({"audio": ["2", 0]})
        self.assertNotIn("1", result)
        self.assertNotIn("2", result)
        self.assertIn("5", result)
        self.assertNotIn("audio", result["5"]["inputs"])

    def test_flat_link_to_kept_node_is_untouched(self):
        result = self._strip({"model": ["40", 0]})
        self.assertEqual(["40", 0], result["5"]["inputs"]["model"])

    def test_nested_link_container_drops_dangling_element_only(self):
        result = self._strip({"wires": [["2", 0], ["40", 1]]})
        self.assertEqual([["40", 1]], result["5"]["inputs"]["wires"])

    def test_nested_link_container_fullly_dangling_deletes_input(self):
        result = self._strip({"wires": [["2", 0], ["1", 0]]})
        self.assertNotIn("wires", result["5"]["inputs"])

    def test_primitive_values_are_untouched(self):
        result = self._strip({"path": "song.mp3", "gain": 3.0})
        self.assertEqual("song.mp3", result["5"]["inputs"]["path"])
        self.assertEqual(3.0, result["5"]["inputs"]["gain"])

    def test_empty_list_input_is_untouched(self):
        result = self._strip({"opts": []})
        self.assertIn("opts", result["5"]["inputs"])
        self.assertEqual([], result["5"]["inputs"]["opts"])

    def test_single_element_list_to_removed_node_deletes_input(self):
        result = self._strip({"one": ["2"]})
        self.assertNotIn("one", result["5"]["inputs"])

    def test_string_option_container_is_untouched(self):
        result = self._strip({"interp": ["linear", "exponential"]})
        self.assertEqual(["linear", "exponential"], result["5"]["inputs"]["interp"])


if __name__ == "__main__":
    unittest.main()
