# Third-party notices

FeverSlop source code is released under the MIT License. The following assets,
workflows, model families, and external projects are separate works and are
not relicensed by FeverSlop. The repository contains workflow JSON and
integration code, not the model weights listed below.

## Workflow and custom-node sources

| Project | Use in FeverSlop | License/source |
| --- | --- | --- |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | External workflow execution service and node runtime | GPL-3.0; see the upstream repository |
| [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) | Basis for several ComfyUI workflow graphs, especially MiniMax H3 workflows | MIT; preserve upstream notices when copying or adapting templates |
| [ComfyUI-Licon-MSR](https://github.com/liconstudio/ComfyUI-Licon-MSR) | MSR workflow/custom-node concepts and LTX MSR integration | MIT; see the upstream repository |

The workflow files under `workflows/` are project configuration and graph
definitions. A workflow's node types may require separately installed ComfyUI
custom nodes with their own licenses.

## Model and adapter sources

Model files must be downloaded from their official sources and are not included
in this repository. The exact filenames and their roles are listed in
[`documentation/workflow-models.md`](documentation/workflow-models.md).

| Model family | Use | License/source |
| --- | --- | --- |
| [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients) | LTX video, audio-conditioning, Ingredients, and reference-sheet workflows | LTX-2 Community License; read the current model terms |
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Text-to-video, reference-to-video, and sequence-reference workflows | MiniMax H3 Community License Agreement; read the current model terms |
| [ACE-Step](https://github.com/ace-step/ACE-Step) | Optional local audio generation workflow | Follow the upstream project and model licenses |
| [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) | Optional video restoration/upscaling workflow | Follow the upstream project and model licenses |
| [SAM3](https://github.com/facebookresearch/sam3) | Optional actor-region segmentation workflow | Follow the upstream project and model terms |
| [InsightFace](https://github.com/deepinsight/insightface) | Face analysis and reference extraction integration | Follow the upstream code, model, and dataset terms |

Image workflow families also reference FLUX.2 Klein, Ideogram 4, Krea2,
Z-Image, SDXL, Qwen, Gemma, and related LoRAs or VAEs. Their license terms
belong to the respective model providers and must be checked before download
or commercial use.

## Python dependencies

Runtime dependencies are declared in `pyproject.toml` and locked in
`uv.lock`. They remain under their own upstream licenses; FeverSlop does not
relicense dependency code. Users distributing a modified or bundled build
should retain the dependency notices required by those projects.
