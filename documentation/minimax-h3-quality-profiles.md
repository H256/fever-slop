# MiniMax H3 quality profiles

All H3 profiles use two passes; a third pass is not supported.
`draft` remains the default for new projects.

| Profile | Pass 1 steps | Pass 2 steps | Pass 2 denoise | Purpose |
|---|---:|---:|---:|---|
| draft | 12 | 4 | 0.55 | fast planning and tests |
| standard | 20 | 8 | 0.40 | balanced production |
| final | 28 | 12 | 0.30 | maximum detail refinement |

These values are reproducible starting calibrations, not hardware-independent
runtime or VRAM guarantees. The exact resolution is determined separately by
the respective resolution profile. In audio profiles, the audio-latent branch
remains unchanged and must not pass through a spatial upscaler.
