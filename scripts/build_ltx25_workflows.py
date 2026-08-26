"""Generate LTX 2.5 profile workflows from maintained API templates."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows" / "video" / "ltx_25"


def _transform(workflow: dict) -> dict:
    result = deepcopy(workflow)
    for node in result.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs") or {}
        if class_type == "UnetLoaderGGUF":
            node["class_type"] = "UNETLoader"
            inputs.pop("unet_name", None)
            inputs["unet_name"] = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
            inputs["weight_dtype"] = "default"
        elif class_type == "DualCLIPLoaderGGUF":
            node["class_type"] = "CLIPLoader"
            inputs.pop("clip_name1", None)
            inputs.pop("clip_name2", None)
            inputs["clip_name"] = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
            inputs["type"] = "ltxv"
        elif class_type == "VAELoaderKJ":
            current = str(inputs.get("vae_name") or "").lower()
            inputs["vae_name"] = (
                "ltx-2.5-audio-vae-bf16.safetensors"
                if "audio" in current else "ltx-2.5-video-vae-conv-bf16.safetensors"
            )
        elif class_type == "LatentUpscaleModelLoader":
            inputs["model_name"] = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
        node["inputs"] = inputs
    def sanitize(value):
        if isinstance(value, dict):
            return {
                key: ("[none]" if key == "lora_name" and "2.3" in str(item).lower() else sanitize(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, str) and "2.3" in value.lower():
            return value.replace("2.3", "2.5").replace("2_3", "2_5")
        return value
    result = sanitize(result)
    return result


def main() -> None:
    templates = {
        "t2v": ROOT / "workflows" / "video_ltxv_i2v_v2.json",
        "i2v": ROOT / "workflows" / "video_ltxv_i2v_v2.json",
        "r2v": ROOT / "workflows" / "video_ltxv_i2v_v2.json",
        "msr": ROOT / "workflows" / "video_default_i2v_ltxv_msr_1actor_1background_v4.json",
        "ingredients": ROOT / "workflows" / "video_ltxv_ingredients_2stage_gguf_v6.json",
    }
    for mode, source in templates.items():
        template = json.loads(source.read_text(encoding="utf-8-sig"))
        for quality in ("draft", "standard", "final"):
            output = OUT / mode / f"{mode}_{quality}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(_transform(template), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            output.with_suffix(".profile.json").write_text(json.dumps({
                "profile_id": f"ltx25-{mode}-{quality}",
                "model_version": "2.5",
                "mode": mode,
                "quality": quality,
                "pass_strategy": "two_pass",
                "audio_policy": "native_audio_when_declared",
                "anchor_policy": "start_frame_and_optional_end_frame" if mode in {"i2v", "r2v"} else "none",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
