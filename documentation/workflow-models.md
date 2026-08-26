# Workflow model requirements

This document records the model references found in the JSON workflows under
`workflows/`, including the historical files under `workflows/old/`. The
inventory was generated from the workflow node inputs at the current release
candidate commit and contains 43 distinct model-file references. It describes
filenames and roles; it does not redistribute any model weights.

ComfyUI resolves these names relative to its model directories. The path and
capitalization in a workflow may therefore matter. A value called a
"checkpoint" in casual usage is not always a monolithic checkpoint: several
workflows load a UNet/diffusion model, VAE, and text encoder separately.

## Audio generation

Used by `audio_song_v2.json` and the historical `old/audio_song.json`:

| Reference | Loader/input | Role |
| --- | --- | --- |
| `acestep_v1.5_turbo.safetensors` | `UNETLoader.tunet_name` | ACE-Step diffusion/UNet model |
| `ace_1.5_vae.safetensors` | `VAELoader.tvae_name` | ACE-Step audio VAE |
| `qwen_0.6b_ace15.safetensors` | `DualCLIPLoader.tclip_name1` | ACE-Step text encoder 1 |
| `qwen_4b_ace15.safetensors` | `DualCLIPLoader.tclip_name2` | ACE-Step text encoder 2 |

## MiniMax H3 two-pass generation

Die Zwei-Pass-Profile trennen den AV-Latent mit den eingebauten ComfyUI-Nodes
`LTXVSeparateAVLatent` und `LTXVConcatAVLatent`. Dadurch bleibt der Audio-Latent
unverändert und wird nicht durch den räumlichen Upscaler geführt. Für das
neuronale H3-Upscaling ist zusätzlich die H3-spezifische Node
`MinimaxH3LatentUpscaler3D` sowie das Modell
`minimax_h3_latent_upscaler_3d_bf16.safetensors` erforderlich. Die Profile
benötigen keine VRGDG-H3-Wrapper-Nodes mehr. Der Upscaler unterstützt in den
Workflows die Betriebsart `scale by multiplier` mit dem bestehenden
`#LATENT_UPSCALE_SCALE`-Anker; Zielgröße, Ausrichtung, Gerät und Präzision sind
explizit im Node hinterlegt.

## LTX video generation

The LTX workflows use separate components. The following references occur in
the current LTX I2V, MSR, Ingredients, and face-fix workflows, as well as
their historical variants:

| Reference | Loader/input | Role |
| --- | --- | --- |
| `LTX-2.3-22B-distilled-1.1-Q6_K.gguf` | `UnetLoaderGGUF.tunet_name` | LTX-2.3 distilled video diffusion model (GGUF) |
| `LTX23_video_vae_bf16.safetensors` | `VAELoaderKJ.tvae_name` | LTX video VAE |
| `LTX23_audio_vae_bf16.safetensors` | `VAELoaderKJ.tvae_name` | LTX audio VAE for audio-conditioned workflows |
| `gemma-3-12b-it-abliterated-sikaworld-high-fidelity-edition.safetensors` | `DualCLIPLoaderGGUF.clip_name1` / `LTXAVTextEncoderLoader.text_encoder` | LTX text encoder |
| `ltx-2.3_text_projection_bf16.safetensors` | `DualCLIPLoaderGGUF.clip_name2` | LTX text projection |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `LatentUpscaleModelLoader.model_name` | LTX latent 2x spatial upscaler |
| `ltx-2.3-22b-dev-fp8.safetensors` | `CheckpointLoaderSimple.ckpt_name` / `LTXAVTextEncoderLoader.ckpt_name` / `LTXVAudioVAELoader.ckpt_name` | LTX development checkpoint and audio-conditioned pipeline component |

LTX LoRAs and adapters:

| Reference | Loader/input | Role |
| --- | --- | --- |
| `LTXV2/ltx23-kwh-balanced-av.comfy.safetensors` | `LoraLoaderModelOnly.lora_name` | LTX audio/video conditioning LoRA |
| `LTXV 2.3/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` | `LoraLoaderModelOnly.lora_name` | LTX video LoRA used by the current video workflows |
| `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` | `LoraLoaderModelOnly.lora_name` | LTX Ingredients/distilled conditioning LoRA |
| `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors` | `LTXICLoRALoaderModelOnly.lora_name` | LTX Ingredients reference-sheet IC-LoRA |
| `LTX-2.3-Licon-MSR-V1.safetensors` | `LTXICLoRALoaderModelOnly.lora_name` | LTX MSR multi-subject-reference LoRA (historical workflows) |
| `LTX-2.3-Licon-MSR-V2.safetensors` | `LTXICLoRALoaderModelOnly.lora_name` | LTX MSR multi-subject-reference LoRA (current workflows) |

## Image generation and start-frame workflows

| Reference | Loader/input | Role | Workflow family |
| --- | --- | --- | --- |
| `z_image_turbo_bf16.safetensors` | `UNETLoader.tunet_name` | Z-Image Turbo diffusion model | `image_t2i_startframe_v1.json`, historical Z-Image workflow |
| `qwen_3_4b.safetensors` | `CLIPLoader.clip_name` | Z-Image text encoder | same as above |
| `ae.safetensors` | `VAELoader.vae_name` | Z-Image VAE | same as above |
| `zimage\own\klw251209-v1_000001250.safetensors` | `LoraLoaderModelOnly.lora_name` | Z-Image LoRA | same as above |
| `flux-2-klein-base-9b-fp8.safetensors` | `UNETLoader.tunet_name` | FLUX.2 Klein base diffusion model | `image_edit_flux2_klein_1ref_v1.json`, `image_edit_flux2_klein_2ref_v1.json` |
| `qwen_3_8b_fp8mixed.safetensors` | `CLIPLoader.clip_name` | FLUX.2 Klein text encoder | same as above |
| `full_encoder_small_decoder.safetensors` | `VAELoader.vae_name` | FLUX.2 Klein VAE | same as above |
| `ideogram4_fp8_scaled.safetensors` | `UNETLoader.tunet_name` | Ideogram 4 diffusion model | `image_t2i_startframe_ideogram*.json` |
| `ideogram4_unconditional_fp8_scaled.safetensors` | `UNETLoader.tunet_name` | Ideogram 4 unconditional model branch | same as above |
| `qwen3vl_8b_fp8_scaled.safetensors` | `CLIPLoader.clip_name` | Ideogram 4 text/vision encoder | same as above |
| `flux2-vae.safetensors` | `VAELoader.vae_name` | Ideogram/FLUX-compatible VAE | same as above |
| `Realism_Engine_Ideogram_V5.safetensors` | `LoraLoaderModelOnly.lora_name` | Ideogram realism LoRA | same as above |
| `krea2_turbo_fp8_scaled.safetensors` | `UNETLoader.tunet_name` | Krea2 Turbo diffusion model | `image_t2i_startframe_krea_v1.json` |
| `qwen3vl_4b_fp8_scaled.safetensors` | `CLIPLoader.clip_name` | Krea2 text/vision encoder | same as above |
| `qwen_image_vae.safetensors` | `VAELoader.vae_name` | Krea2 VAE | same as above |
| `SDXL 1.0\realistic\realvisxlV50_v50Bakedvae.safetensors` | `CheckpointLoaderSimple.ckpt_name` / Easy-Use loader | SDXL checkpoint | `image_repair_sdxl_ipadapter_identity_v1.json`, `image_detail_easyuse_startframe_v1.json` |

Additional image-analysis models:

| Reference | Loader/input | Role |
| --- | --- | --- |
| `sam3.safetensors` | `easy sam3ModelLoader.model` | SAM3 segmentation model |
| `sam_vit_b_01ec64.pth` | `easy samLoaderPipe.model_name` | SAM ViT-B segmentation model |
| `bbox/face_yolov8n.pt` | `easy ultralyticsDetectorPipe.model_name` | YOLOv8 face detector |

## MiniMax H3 video and reference generation

Used by `video_minimax_h3_*.json` and
`sequence_to_sheet_minimax_h3_i2va_v1.json`:

| Reference | Loader/input | Role |
| --- | --- | --- |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `UNETLoader.tunet_name` | MiniMax H3 reference-to-video diffusion model |
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `UNETLoader.tunet_name` | MiniMax H3 first/last-frame or text-to-video diffusion model |
| `minimax_h3_video_vae_fp16.safetensors` | `VAELoader.vae_name` | MiniMax H3 video VAE |
| `minimax_h3_audio_vae_fp32.safetensors` | `VAELoader.vae_name` | MiniMax H3 audio VAE |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `CLIPLoader.clip_name` | MiniMax H3 text/vision encoder |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `LoraLoaderModelOnly.lora_name` | MiniMax H3 fast reference/sequence LoRA |

## SeedVR2 video upscaling

Used by `video_seedvr2_3b_api.json`:

| Reference | Loader/input | Role |
| --- | --- | --- |
| `seedvr2_3b_int8_convrot.safetensors` | `UNETLoader.tunet_name` | SeedVR2 3B video restoration/upscaling model |
| `seedvr2_ema_vae_fp16.safetensors` | `VAELoader.vae_name` | SeedVR2 VAE |

## Non-model file references

The workflow inventory also found `LoadAudio`, `LoadVideo`, and output-prefix
values such as `ComfyUI_00056_.mp3`, `The Well Of Youth.wav`, and
`ComfyUI_00103.mp3`. These are input/output assets or saved workflow state,
not model requirements. Production runs should replace them through the
pipeline's workflow patching and asset-upload logic.

## Licensing and provenance

The filenames above identify external model assets. They are not covered by
FeverSlop's MIT license. Users must obtain each model from its official source
and follow its applicable license and usage terms. In particular, the LTX and
MiniMax model families use model-specific community licenses. See
`THIRD_PARTY_NOTICES.md` when the public release attribution inventory is
completed.
