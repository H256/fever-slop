import unittest

from feverslop.domain.performance_timeline import project_performance


class PerformanceTimelineTests(unittest.TestCase):
    def vocal(self, words):
        return {"start": 50.55, "end": 58.6, "type": "vocals", "lyrics": "hold me",
                "word_timestamps": words}

    def test_measured_onset_and_short_internal_pause(self):
        phases = project_performance([self.vocal([
            {"word": "hold", "start": 52.7, "end": 54, "source": "whisper", "word_id": "w1"},
            {"word": "me", "start": 54.1, "end": 57, "source": "corrected_from_whisper", "word_id": "w2"},
        ])], 50.55, 58.6)
        self.assertEqual([(p["start"], p["end"], p["state"]) for p in phases], [
            (50.55, 52.7, "instrumental"), (52.7, 54, "singing"),
            (54, 54.1, "instrumental"), (54.1, 57, "singing"), (57, 58.6, "instrumental")])
        self.assertTrue(all(p["acoustically_verified"] for p in phases))

    def test_cut_word_continues_without_duplicating_text(self):
        timeline = [self.vocal([{"word": "hold", "start": 52, "end": 56,
                                 "source": "whisper", "word_id": "w1"}])]
        left = project_performance(timeline, 52, 53)[0]
        right = project_performance(timeline, 53, 56)[0]
        self.assertEqual(left["state"], "singing")
        self.assertEqual(left["lyrics"], "")
        self.assertEqual(right["lyrics"], "hold")
        self.assertEqual(left["word_timestamps"][0]["end"], 53)
        self.assertEqual(right["word_timestamps"][0]["start"], 53)
        self.assertEqual(left["word_timestamps"][0]["continuation_of_word"], "w1")
        self.assertEqual(right["word_timestamps"][0]["word_id"], "w1")

    def test_legacy_timing_is_unverified(self):
        phase = project_performance([self.vocal([])], 51, 52)[0]
        self.assertFalse(phase["acoustically_verified"])
        self.assertIn("legacy_timing_unverified", phase["reason_codes"])
        self.assertTrue(phase["performance_conflicts"])

    def test_two_voices_retain_metadata(self):
        timeline = [dict(self.vocal([{"word": "hold", "start": 52, "end": 56,
            "source": "whisper", "word_id": "w1"}]), speaker_id="voice1", offscreen=True),
            dict(self.vocal([{"word": "me", "start": 52, "end": 56,
            "source": "whisper", "word_id": "w2"}]), speaker_id="voice2")]
        phase = project_performance(timeline, 52, 56)[0]
        self.assertEqual([v["speaker_id"] for v in phase["vocal_sources"]], ["voice1", "voice2"])
        self.assertTrue(phase["vocal_sources"][0]["offscreen"])

    def test_empty_evidence_does_not_verify_silence(self):
        phase = project_performance([], 0, 2)[0]
        self.assertFalse(phase["acoustically_verified"])
        self.assertIn("missing_performance_evidence", phase["reason_codes"])

    def test_unresolved_alignment_retains_diagnostic_in_silent_gap(self):
        timeline = [dict(self.vocal([]), alignment={"targets": [{"word": "hold", "source": "unresolved"}], "timed_words": []})]
        phase = project_performance(timeline, 51, 52)[0]
        self.assertIn("unresolved_lyric_alignment", phase["reason_codes"])

    def test_simultaneous_events_assign_each_voice_only_its_own_text(self):
        timeline = [dict(self.vocal([{"word": word, "start": 52, "end": 56,
            "source": "whisper", "word_id": voice}]), speaker_id=voice)
            for word, voice in (("hold", "voice1"), ("me", "voice2"))]
        phase = project_performance(timeline, 52, 56)[0]
        self.assertEqual(["hold", "me"], [event["lyrics"] for event in phase["vocal_events"]])
        self.assertNotIn("speaker_id", phase)

    def test_word_level_speakers_remain_separate_events(self):
        timeline = [self.vocal([
            {"word": "hold", "start": 52, "end": 56, "source": "whisper", "word_id": "a", "speaker_id": "voice1"},
            {"word": "me", "start": 52, "end": 56, "source": "whisper", "word_id": "b", "speaker_id": "voice2", "offscreen": True},
        ])]
        phase = project_performance(timeline, 52, 56)[0]
        self.assertEqual(["voice1", "voice2"], [event["speaker_id"] for event in phase["vocal_events"]])
        self.assertEqual(["hold", "me"], [event["lyrics"] for event in phase["vocal_events"]])
        self.assertTrue(phase["vocal_events"][1]["offscreen"])

    def test_explicit_instrumental_entry_verifies_covered_silence(self):
        for key in ("type", "kind"):
            phase = project_performance([{key: "instrumental", "start": 0, "end": 4}], 0, 2)[0]
            self.assertEqual("instrumental", phase["state"])
            self.assertTrue(phase["acoustically_verified"])
            self.assertEqual([], phase["performance_conflicts"])

    def test_gap_between_instrumental_entries_is_not_verified(self):
        phases = project_performance([
            {"type": "instrumental", "start": 0, "end": 1},
            {"type": "instrumental", "start": 2, "end": 3},
        ], 0, 3)
        self.assertEqual([(0, 1), (1, 2), (2, 3)], [(p["start"], p["end"]) for p in phases])
        self.assertEqual([True, False, True], [p["acoustically_verified"] for p in phases])
        self.assertIn("missing_performance_evidence", phases[1]["reason_codes"])

    def test_outside_vocal_envelope_is_not_verified(self):
        phases = project_performance([self.vocal([
            {"word": "hold", "start": 52, "end": 56, "source": "whisper", "word_id": "w1"},
        ])], 49, 60)
        self.assertFalse(phases[0]["acoustically_verified"])
        self.assertFalse(phases[-1]["acoustically_verified"])

    def test_uncertain_instrumental_evidence_does_not_certify_silence(self):
        phase = project_performance([{"type": "instrumental", "start": 0, "end": 2,
            "evidence": {"activity_status": "uncertain"}}], 0, 2)[0]
        self.assertFalse(phase["acoustically_verified"])
