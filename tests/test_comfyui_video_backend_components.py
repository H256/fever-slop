import unittest
from pathlib import Path


class FakeComfyUIClient:
    def __init__(self):
        self.audio_uploads = []
        self.image_uploads = []

    def upload_file_via_image_endpoint(self, path, *, subfolder, file_type, overwrite):
        self.audio_uploads.append((Path(path), subfolder, file_type, overwrite))
        return {"name": Path(path).name, "subfolder": subfolder}

    def upload_image(self, path, *, subfolder, file_type, overwrite):
        self.image_uploads.append((Path(path), subfolder, file_type, overwrite))
        return {"name": Path(path).name, "subfolder": subfolder}


class ComfyUIVideoAssetUploaderTests(unittest.TestCase):
    def test_upload_audio_uses_comfyui_image_endpoint_contract(self):
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        client = FakeComfyUIClient()
        uploader = ComfyUIVideoAssetUploader(client)

        name = uploader.resolve_audio_name(
            Path("song.mp3"),
            upload_audio=True,
            uploaded_audio_name=None,
        )

        self.assertEqual("autoprompter/audio/song.mp3", name)
        self.assertEqual(
            [(Path("song.mp3"), "autoprompter/audio", "input", True)],
            client.audio_uploads,
        )

    def test_audio_upload_can_be_skipped_with_uploaded_name_or_file_name(self):
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        uploader = ComfyUIVideoAssetUploader(FakeComfyUIClient())

        self.assertEqual(
            "already/uploaded.mp3",
            uploader.resolve_audio_name(
                Path("song.mp3"),
                upload_audio=False,
                uploaded_audio_name="already/uploaded.mp3",
            ),
        )
        self.assertEqual(
            "song.mp3",
            uploader.resolve_audio_name(
                Path("song.mp3"),
                upload_audio=False,
                uploaded_audio_name=None,
            ),
        )

    def test_upload_startframe_uses_storyboard_subfolder_contract(self):
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        client = FakeComfyUIClient()
        uploader = ComfyUIVideoAssetUploader(client)

        name = uploader.resolve_startframe_name(
            Path("scene_0001.png"),
            upload_startframes=True,
        )

        self.assertEqual("autoprompter/storyboard/scene_0001.png", name)
        self.assertEqual(
            [(Path("scene_0001.png"), "autoprompter/storyboard", "input", True)],
            client.image_uploads,
        )

    def test_startframe_upload_can_be_skipped(self):
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        uploader = ComfyUIVideoAssetUploader(FakeComfyUIClient())

        self.assertEqual(
            "scene_0001.png",
            uploader.resolve_startframe_name(
                Path("scene_0001.png"),
                upload_startframes=False,
            ),
        )

    def test_malformed_upload_response_raises_clear_error(self):
        from autoprompter.adapters.comfyui_video_assets import ComfyUIVideoAssetUploader

        with self.assertRaisesRegex(ValueError, "Unexpected ComfyUI upload response"):
            ComfyUIVideoAssetUploader.comfy_path_from_upload({"subfolder": "x"})


if __name__ == "__main__":
    unittest.main()
