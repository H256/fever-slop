from pathlib import Path
import unittest


from feverslop.domain.h3_audio_delivery import load_h3_audio_delivery


class H3AudioDeliveryTests(unittest.TestCase):
    def test_audio_latent_profile_declares_generation_conditioning(self):
        workflow = Path("workflows/video/minimax_h3/r2v_audio_two_pass.json")

        delivery = load_h3_audio_delivery(workflow)

        self.assertTrue(delivery.conditions_generation)
        self.assertTrue(delivery.copies_to_output)
        self.assertFalse(delivery.is_audience_only_music)
        self.assertEqual("preserve_original_av_audio_latent", delivery.audio_policy)

    def test_non_audio_profile_does_not_claim_audio_conditioning(self):
        workflow = Path("workflows/video/minimax_h3/r2v_two_pass.json")

        delivery = load_h3_audio_delivery(workflow)

        self.assertFalse(delivery.conditions_generation)
        self.assertFalse(delivery.copies_to_output)
        self.assertFalse(delivery.is_audience_only_music)




class ResolvedAudioSourceTests(unittest.TestCase):
    @staticmethod
    def graph():
        return {
            '1': {'class_type': 'LoadAudio', '_meta': {'title': '#AUDIO_1'}, 'inputs': {'audio': 'old.wav'}},
            '2': {'class_type': 'TrimAudioDuration', '_meta': {'title': '#TRIM_AUDIO_1'}, 'inputs': {'audio': ['1', 0], 'start_index': 0, 'duration': 1}},
            '3': {'class_type': 'LoadAudio', '_meta': {'title': '#AUDIO_2'}, 'inputs': {'audio': 'old2.wav'}},
            '4': {'class_type': 'TrimAudioDuration', '_meta': {'title': '#TRIM_AUDIO_2'}, 'inputs': {'audio': ['3', 0], 'start_index': 0, 'duration': 1}},
            '5': {'class_type': 'MiniMaxH3AddGuide', 'inputs': {'audio': ['2', 0]}},
            '6': {'class_type': 'MiniMaxH3ReferenceToVideo', 'inputs': {}},
        }

    def resolve(self, refs, source='selected_performer', graph=None):
        from feverslop.domain.h3_audio_delivery import H3AudioDelivery, resolve_h3_audio_sources
        from feverslop.domain.audio_timing_contract import AudioTimingWindow
        return resolve_h3_audio_sources(
            H3AudioDelivery(conditioning_source=source), refs, AudioTimingWindow(12.5, 16.75),
            workflow=self.graph() if graph is None else graph,
        )

    def test_full_mix_conditioning_is_independent_of_reference_order(self):
        refs = [{'source': 'drums.wav', 'name': 'drums'}, {'source': 'mix.wav', 'name': 'full_mix'}]
        for ordered in (refs, list(reversed(refs))):
            sources = self.resolve(ordered, 'full_mix')
            conditioning = [s for s in sources if 'conditioning' in s['roles']]
            self.assertEqual(['mix.wav'], [s['source_path'] for s in conditioning])
            self.assertEqual({'start_seconds': 12.5, 'end_seconds': 16.75}, conditioning[0]['audio_timing_window'])

    def test_legacy_guide_tracks_actual_first_slot(self):
        sources = self.resolve([{'source': 'drums.wav', 'name': 'drums'}, {'source': 'mix.wav', 'name': 'full_mix'}])
        self.assertIn('conditioning', sources[0]['roles'])
        self.assertNotIn('conditioning', sources[1]['roles'])

    def test_unknown_guide_chain_is_machine_readable_error(self):
        from feverslop.domain.h3_audio_delivery import H3AudioContractError
        graph = self.graph()
        graph['2']['class_type'] = 'UnknownAudioTransform'
        with self.assertRaises(H3AudioContractError) as caught:
            self.resolve([{'source': 'drums.wav'}], graph=graph)
        self.assertEqual('h3_audio_unknown_topology', caught.exception.code)

    def test_reference_only_and_no_audio_have_no_conditioning(self):
        graph = self.graph()
        del graph['5']
        self.assertEqual(['reference'], self.resolve([{'source': 'drums.wav'}], graph=graph)[0]['roles'])
        self.assertEqual([], self.resolve([], graph={}))

    def test_explicit_condition_without_guide_is_rejected(self):
        from feverslop.domain.h3_audio_delivery import H3AudioContractError
        with self.assertRaises(H3AudioContractError):
            self.resolve([{'source': 'mix.wav', 'name': 'full_mix'}], 'full_mix', graph={})

class AudioGraphValidationTests(unittest.TestCase):
    def test_direct_output_copy_has_separate_binding(self):
        graph = ResolvedAudioSourceTests.graph()
        graph['7'] = {'class_type': 'VHS_VideoCombine', 'inputs': {'audio': ['4', 0]}}
        sources = ResolvedAudioSourceTests().resolve([{'source': 'drums.wav'}, {'source': 'mix.wav'}], graph=graph)
        self.assertNotIn('output_copy', sources[0]['roles'])
        self.assertIn('output_copy', sources[1]['roles'])

    def test_declared_preserved_latent_requires_actual_edge(self):
        from feverslop.domain.h3_audio_delivery import H3AudioDelivery, H3AudioContractError, resolve_h3_audio_sources
        with self.assertRaises(H3AudioContractError) as caught:
            resolve_h3_audio_sources(H3AudioDelivery(audio_policy='preserve_original_av_audio_latent'), [], None, workflow={})
        self.assertEqual('h3_audio_profile_contradiction', caught.exception.code)

    def test_postpatch_wrong_source_and_trim_rejected(self):
        from feverslop.domain.h3_audio_delivery import H3AudioContractError, apply_h3_audio_sources, validate_h3_audio_sources
        graph = ResolvedAudioSourceTests.graph()
        sources = ResolvedAudioSourceTests().resolve([{'source': 'drums.wav'}], graph=graph)
        graph['6']['inputs']['ref_audios.ref_audio_0'] = ['2', 0]
        apply_h3_audio_sources(graph, sources)
        with self.assertRaises(H3AudioContractError) as caught:
            validate_h3_audio_sources(graph, sources, {'drums.wav': 'drums-upload.wav'})
        self.assertEqual('h3_audio_source_mismatch', caught.exception.code)
        graph['1']['inputs']['audio'] = 'drums-upload.wav'
        with self.assertRaises(H3AudioContractError) as caught:
            validate_h3_audio_sources(graph, sources, {'drums.wav': 'drums-upload.wav'})
        self.assertEqual('h3_audio_timing_mismatch', caught.exception.code)

class AudioFingerprintTests(unittest.TestCase):
    def test_graph_change_updates_fingerprint_without_profile(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'workflow.json'
            path.write_text('{}')
            before = load_h3_audio_delivery(path)
            path.write_text('{"1": {}}')
            after = load_h3_audio_delivery(path)
            self.assertNotEqual(before.workflow_hash, after.workflow_hash)
            self.assertEqual(str(path), before.workflow_path)
            self.assertEqual(before, type(before).from_context(before.to_context()))


if __name__ == "__main__":
    unittest.main()
