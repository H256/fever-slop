#!/usr/bin/env python3
"""r2v_prompt_check.py — Validate R2V prompt generation against a real LLM.

Usage:
    uv run python tools/r2v_prompt_check.py
    uv run python tools/r2v_prompt_check.py --pipeline ltx_i2v   # base mode (3-field)
    uv run python tools/r2v_prompt_check.py --model gemma4-31b   # override model
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load_api_key(cfg_path: pathlib.Path) -> str:
    """Load LLM API key from app_config.json."""
    cfg = json.loads(cfg_path.read_text())
    llm_cfg = cfg.get("llm", {})
    key = llm_cfg.get("api_key")
    if key and key not in (None, "not-needed"):
        return key
    raise SystemExit("No API key in app_config.json (llm.api_key)")


def main():
    parser = argparse.ArgumentParser(description="R2V prompt checker")
    parser.add_argument("--pipeline", default="minimax-h3-r2v",
                        help="Video pipeline (default: minimax-h3-r2v)")
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--base-url", default=None, help="Override LLM base URL")
    parser.add_argument("--config", default=str(pathlib.Path(__file__).resolve().parent.parent / "app_config.json"),
                        help="Path to app_config.json")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    api_key = load_api_key(cfg_path)

    # Load runtime config
    cfg = json.loads(cfg_path.read_text())
    llm_cfg = cfg.get("llm", {})
    base_url = args.base_url or llm_cfg.get("base_url", "http://your-llm-server.local/v1")
    model = args.model or llm_cfg.get("model", "gemma4-26b-a4b")

    # Import project modules
    from feverslop.adapters.llm_client import LocalOpenAIClient
    from feverslop.domain.llm_parsing import extract_json_object
    from feverslop.prompting.h3_prompt_builder import build_references_from_segment
    from feverslop.prompting.minimax_h3_prompt_style import build_h3_video_system_prompt

    llm = LocalOpenAIClient(base_url=base_url, model=model, api_key=api_key, max_tokens=2048)
    print(f"🔌 LLM: {model} @ {base_url}")

    # ─── Demo segment ───────────────────────────────────────────────
    segment: dict = {
        "segment_id": "voc_A",
        "type": "vocals",
        "time_start": 0,
        "time_end": 30,
        "concept": "Bob the squirrel and Laila the rat lady are seated in a cozy cafe. Bob is excitedly talking about nuts, while Laila reacts with clear distaste.",
        "timeline": "verse 1",
        "references": {
            "reference_image_paths": [
                "projects/test-movie/output/ref/stage_3/msr/actors/H256_Bob.png",
                "projects/test-movie/output/ref/stage_3/msr/actors/H256_Laila.png",
                "projects/test-movie/output/ref/stage_3/msr/locations/Cafe.png",
            ],
            "reference_audio_paths": [
                "projects/test-movie/song.mp3",
                "projects/test-movie/sfx/cafe-ambience.wav",
            ],
        },
        "ref_items": [
            {"type": "actor", "name": "Bob", "visual_description": "obese squirrel"},
            {"type": "actor", "name": "Laila", "visual_description": "old eccentric rat lady"},
            {"type": "location", "name": "Cafe", "visual_description": "cozy interior"},
        ],
    }

    # ─── Resolve references ────────────────────────────────────────
    refs = build_references_from_segment(segment)
    print(f"\n📋 References: {json.dumps(refs, indent=2)}")

    # ─── Resolve mode from pipeline ────────────────────────────────
    mode = "ref" if args.pipeline == "minimax-h3-r2v" else "base"
    print(f"🎬 Mode: {mode} (pipeline: {args.pipeline})")

    # ─── Build system prompt ───────────────────────────────────────
    prompt = build_h3_video_system_prompt(
        mode=mode,
        video_type="music_video",
        silent_mode=False,
        references=refs if mode == "ref" else None,
    )

    # Show relevant sections
    print("\n--- System Prompt Highlights ---")
    for line in prompt.split("\n"):
        if any(tok in line for tok in ("Reference Labels Used", "Mandatory", "MANDATORY", "audio=fully_preserved", "Audio Preservation")):
            print(line)

    # ─── Build user payload ────────────────────────────────────────
    user_payload = {
        "scene_concept": segment["concept"],
        "camera_motion": "static",
        "character_motion": "Bob talking, Laila reacting",
        "timeline_context": segment["timeline"],
        "subject": "Bob the squirrel and Laila the rat lady",
        "story_idea": "Bob excitedly talks about nuts; Laila reacts with distaste",
        "style": "Cinematic 8k, highly detailed",
        "locations": ["cafe"],
        "vocal_track": "present",
        "silent_mode": False,
        "audio_content": "lyrics",
        "location_constraint": "",
    }

    # ─── Call LLM ──────────────────────────────────────────────────
    print("\n🤖 Calling LLM...")
    result = llm.complete_prompt(system_prompt=prompt, prompt=json.dumps(user_payload))

    # ─── Parse and validate ────────────────────────────────────────
    try:
        parsed = extract_json_object(result)
    except Exception as e:
        print(f"\n❌ Parse failed: {e}")
        print("Raw output:")
        print(result[:500])
        sys.exit(1)

    expected_keys = ["subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"]
    if mode == "base":
        # base mode only expects 3 fields
        expected_keys = ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"]

    missing = [k for k in expected_keys if k not in parsed]
    print(f"\n{'✅' if not missing else '❌'} Fields present: {list(parsed.keys())}")
    if missing:
        print(f"   Missing: {missing}")

    # Audio validation (ref mode)
    if mode == "ref" and refs and refs.get("audio", []):
        sd = parsed.get("subject_definitions", "")
        ra = parsed.get("retention_analysis", "")
        audio_in_sd = "<Audio" in sd
        audio_in_ra = "<Audio" in ra
        print(f"\n{'✅' if audio_in_sd else '❌'} Audio in subject_definitions")
        print(f"{'✅' if audio_in_ra else '❌'} Audio in retention_analysis")
    else:
        print("\n   (no audio refs to check)")

    # ─── Pretty output ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MERGED PROMPT (ComfyUI #PROMPT)")
    print("=" * 60)

    if mode == "ref":
        for field in ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"):
            val = parsed.get(field, "")
            if val:
                print(f"\n{field}:")
                print(val)
    else:
        for field in ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"):
            val = parsed.get(field, "")
            if val:
                print(f"\n{field}:")
                print(val)


if __name__ == "__main__":
    main()
