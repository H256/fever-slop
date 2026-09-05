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


if __name__ == "__main__":
    unittest.main()
