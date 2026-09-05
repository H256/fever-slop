"""Shared seed policy for ComfyUI scene rendering."""

import random


def resolve_scene_seed(seed_offset: int, randomize_seed: bool, scene: int | dict) -> int:
    if randomize_seed:
        return random.randint(0, 2**63 - 1)
    if isinstance(scene, dict) and scene.get("seed") is not None:
        return int(scene["seed"])
    scene_number = int(scene.get("scene", 0)) if isinstance(scene, dict) else int(scene)
    return seed_offset + scene_number
