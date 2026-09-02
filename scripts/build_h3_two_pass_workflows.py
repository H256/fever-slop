"""Build native MiniMax H3 two-pass API workflow profiles from local templates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "workflows" / "video" / "minimax_h3"


def _node(workflow: dict, class_type: str, title: str, inputs: dict) -> str:
    numeric = [int(key) for key in workflow if str(key).isdigit()]
    node_id = str(max(numeric, default=0) + 1)
    workflow[node_id] = {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}
    return node_id


def _first(workflow: dict, class_type: str) -> tuple[str, dict]:
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return str(node_id), node
    raise KeyError(f"missing workflow node: {class_type}")


def build(template: Path, output: Path, *, audio: bool = False) -> None:
    workflow = json.loads(template.read_text(encoding="utf-8-sig"))
    sampler_id, sampler = _first(workflow, "SamplerCustomAdvanced")
    sampler.setdefault("_meta", {})["title"] = "#PASS1"
    model_id, _ = _first(workflow, "UNETLoader")
    vae_id, _ = _first(workflow, "VAELoader")
    h3_candidates = ("MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo", "MiniMaxH3Video")
    h3_id, _ = next(
        (_first(workflow, class_type) for class_type in h3_candidates if any(
            node.get("class_type") == class_type for node in workflow.values()
        )),
    )
    if audio:
        _, audio_decode = _first(workflow, "VAEDecodeAudio")
        audio_decode.setdefault("_meta", {})["title"] = "#AUDIO_LATENT"

    pass1_sampler = _node(workflow, "KSamplerSelect", "#PASS1_SAMPLER", {"sampler_name": "res_multistep"})
    pass1_scheduler = _node(workflow, "BasicScheduler", "#PASS1_SCHEDULER", {
        "model": [model_id, 0], "scheduler": "simple", "steps": 20, "denoise": 1.0,
    })
    sampler_inputs = sampler.setdefault("inputs", {})
    sampler_inputs["sampler"] = [pass1_sampler, 0]
    sampler_inputs["sigmas"] = [pass1_scheduler, 0]
    separate = _node(workflow, "LTXVSeparateAVLatent", "#SEPARATE_AV", {"av_latent": [sampler_id, 0]})
    upscale = _node(workflow, "MinimaxH3LatentUpscaler3D", "#LATENT_UPSCALE", {
        "latent": [separate, 0],
        "model_name": "minimax_h3_latent_upscaler_3d_bf16.safetensors",
        "mode": "scale by multiplier",
        # ComfyUI serializes the mode-dependent multiplier under the
        # mode.scale API input name, not as a standalone scale input.
        "mode.scale": 2.0,
        "align": 32,
        "enable_chunking": False,
        "force_unload": True,
        "enable_temporal_chunking": True,
        "device": "cuda",
        "precision": "bf16",
    })
    recombine = _node(workflow, "LTXVConcatAVLatent", "#RECOMBINE_AV", {
        "video_latent": [upscale, 0], "audio_latent": [separate, 1],
    })
    pass2_sampler_select = _node(workflow, "KSamplerSelect", "#PASS2_SAMPLER", {"sampler_name": "res_multistep"})
    pass2_scheduler = _node(workflow, "BasicScheduler", "#PASS2_SCHEDULER", {
        "model": [model_id, 0], "scheduler": "simple", "steps": 4, "denoise": 0.55,
    })
    guider = _node(workflow, "BasicGuider", "#PASS2_GUIDER", {
        "model": [model_id, 0], "conditioning": [h3_id, 0],
    })
    noise = _node(workflow, "RandomNoise", "#PASS2_NOISE", {"noise_seed": 0})
    pass2 = _node(workflow, "SamplerCustomAdvanced", "#PASS2", {
        "noise": [noise, 0], "guider": [guider, 0], "sampler": [pass2_sampler_select, 0],
        "sigmas": [pass2_scheduler, 0], "latent_image": [recombine, 0],
    })
    decode = _node(workflow, "VAEDecode", "#PASS2_DECODE", {"samples": [pass2, 0], "vae": [vae_id, 0]})
    _, output_node = _first(workflow, "VHS_VideoCombine")
    output_node.setdefault("inputs", {})["images"] = [decode, 0]
    profile = {
        "model_family": "minimax-h3",
        "pass_strategy": "two_pass",
        "topology": ["#PASS1", "#SEPARATE_AV", "#LATENT_UPSCALE", "#RECOMBINE_AV", "#PASS2"],
        "audio_policy": "preserve_original_av_audio_latent" if audio else "not_applicable",
        "preserve_audio_latent": audio,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    build(WORKFLOW_DIR / "r2v_v1.json", WORKFLOW_DIR / "r2v_two_pass.json")
    build(WORKFLOW_DIR / "r2v_audio_v1.json", WORKFLOW_DIR / "r2v_audio_two_pass.json", audio=True)
    build(WORKFLOW_DIR / "t2v.json", WORKFLOW_DIR / "t2v_two_pass.json")


if __name__ == "__main__":
    main()
