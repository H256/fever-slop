"""Shared ComfyUI render port fakes."""

from pathlib import Path


class FakeClient:
    def __init__(self):
        self.uploaded = []
        self.uploaded_paths = []
        self.queued_workflow = None

    def upload_image(self, path, subfolder, file_type, overwrite, upload_name=None):
        name = upload_name or Path(path).name
        self.uploaded.append(name)
        self.uploaded_paths.append(Path(path))
        return {"name": name, "subfolder": subfolder, "type": file_type}

    def upload_file_via_image_endpoint(self, path, subfolder, file_type, overwrite, upload_name):
        return {"name": upload_name, "subfolder": subfolder, "type": file_type}

    def queue_prompt(self, workflow):
        self.queued_workflow = workflow
        return "prompt-id"

    def wait_for_completion(self, prompt_id):
        return {"outputs": {"save": {"videos": [{"filename": "scene.mp4", "type": "output"}]}}}

    def download_view_file(self, filename, subfolder, file_type, output_path):
        return Path(output_path)


class FakeRenderQueue:
    def __init__(self):
        self.calls = []

    def queue_workflow_and_download_first_video(self, workflow, scene_number, output_path):
        self.calls.append({
            "workflow": workflow,
            "scene_number": scene_number,
            "output_path": Path(output_path),
        })
        return Path(output_path)


class FakePostProcessor:
    def __init__(self):
        self.trim_specs = []

    def trim_clip(self, spec):
        self.trim_specs.append(spec)
        return spec.output_file
