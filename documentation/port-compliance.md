# Port-Adapter Compliance Matrix

Status: Planned
Created: 2026-08-01

## Purpose

Maps every port protocol in `src/feverslop/ports/` to its known adapter implementations
under `src/feverslop/adapters/`. Use this table to verify that each port has at least one
production adapter and that test fakes match the same method signatures.

## Ports and Adapters

| Port Protocol | Package | Adapter Implementations | Test Fakes/Mocks |
|---|---|---|---|
| `ArtifactStoragePort` | `ports.storage` | `LocalMovieArtifactWriter`, `JsonArtifactStore`, `ComfyUIVideoAssetUploader` | `FakeArtifactStore` (test_architecture_ports) |
| `ConfigProviderPort` | `ports.configuration` | `ProjectConfigLoader`, `ConfigFileProvider` | `FakeConfigProvider` (test configs) |
| `DemucsSeparatorPort` | `ports.audio` | `DemucsSeparator` | `FakeDemucsSeparator` (movie tests) |
| `FaceDetectionPort` | `ports.facepipeline` | `InsightFaceDetectorAdapter` | — (face tests use real detector or raw inputs) |
| `FaceIdentityPort` | `ports.facepipeline` | `FaceIdentityAdapter` | — |
| `FaceMaskPort` | `ports.facepipeline` | `FaceMaskAdapter` | — |
| `ImageRenderBackend` | `ports.rendering` | `ComfyUIImageBackend`, `LocalMovieImageBackend`, `StoryboardRenderer` | `FakeImageRenderBackend` (test_architecture_ports) |
| `JSONLLMProvider` | `ports.llm` | `OpenRouterLLM`, `GroqLLM`, `OllamaLLM`, `LocalLLMClient` | `FakeLLM` (test_full_auto, test_full_auto_llm_chain) |
| `PromptRevision` | `ports.prompt_revision` | `SqliteRevisionStore` | — (sqlite_adapter tests use live temp-db SQLite) |
| `ProjectScaffoldPort` | `ports.projects` | `LocalProjectScaffold`, `ProjectReferenceLibrary` | `FakeProjectScaffold` (tests) |
| `RevisionStorePort` | `ports.revision` | `SqliteRevisionStore` | `InMemoryRevisionStore` (where available) |
| `SceneStoryboardPort` | `ports.storyboard` | `StoryboardRenderer` | `FakeStoryboardRenderer` (tests) |
| `SongBriefGenerator` | `ports.song_brief` | `LLMSongBriefGenerator` | `FakeSongBriefGenerator` (test_full_auto) |
| `SongGeneratorPort` | `ports.audio` | `LocalSongGenerator`, `MusicgenSongGenerator` | `FakeSongGenerator` (test_full_auto) |
| `TimelineAnalyserPort` | `ports.audio` | `VocalTimelineAnalyzer`, `BeatImpactAnalyzer` | `FakeTimelineAnalyser` (tests) |
| `VideoRenderBackend` | `ports.rendering` | `ComfyUIVideoRenderBackend`, `ComfyUIMSRVideoRenderBackend`, `ComfyUIFaceFixCropBackend`, `ComfyUIFaceFixRenderBackend` (deprecated) | `FakeVideoBackend` (test_architecture_ports, test_cli_to_pipeline_fake_ports) |
| `VideoPostprocessorPort` | `ports.postprocessing` | `VideoPostProcessor`, `PostprocessorFrameExtractor` | — |
| `WorkflowPatcherPort` | `ports.workflow` | `WorkflowPatcher`, `MovieWorkflowPatcher`, `WorkflowMaterializer` | — |
| `RunPipelinePort` | `ports.pipeline` | `RunPipelineAdapter` | `FakeRunner` (test_full_auto) |

## Notes

- **LLMSongBriefGenerator** is the real adapter tested in `test_full_auto_llm_chain.py` — it
  receives a `FakeLLM` but performs real JSON parsing and `SongSpec` construction.
- **RenderVideoScenesUseCase** is the real application service tested in
  `test_cli_to_pipeline_fake_ports.py` — it receives a real `JsonArtifactStore` and
  `FakeVideoBackend`.
- **PatchPromptUseCase** and **LoadPromptHistoryUseCase** are real application services tested
  in `test_studio_rebuild_service.py` via `RebuildServiceUseCaseChainTests` with
  a real `SqliteRevisionStore` backed by a temporary SQLite file.
- Ports marked with `—` in the fakes column have no dedicated test fake; tests either use
  the real adapter directly, raw data bypass, or inline mocks.
- `MovieWorkflowPatcher` is a domain-level utility, not a direct port implementation; it
  wraps `WorkflowPatcher` for movie-specific audio-stripping and scene-configuration logic.
- `LocalMovieI2VEditVisualAdapter` and `LocalMovieVisualAdapter` are local/test adapter
  implementations that do not conform to any single port protocol but provide movie-level
  orchestration (render_movie, render_images, plan, etc.).
- `FaceDebugAdapter` implements the informal `DebugArtifactPort` for writing face-pipeline
  debug PNGs; this is not a formal port in `ports/` yet.
- `encode_face_crop_mp4` in `face_video_encoder.py` is a free function, not a class adapter;
  it implements FFmpeg-based frame encoding.
- `InsightFaceDetectorAdapter` is identified as `FaceDetectorPort` which is part of the
  `ports.face_pipeline` module along with `FaceIdentityPort` and `FaceMaskPort`.
