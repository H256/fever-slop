param(
    [string]$ProjectConfig = ".\projects\my_first_project\config.json",
    [string]$AppConfig = ".\app_config.json",
    [int]$ConceptBatchSize = 10,
    [string]$StoryboardWorkflow = ".\workflows\autoprompt_image_z_image_turbo.json",
    [string]$RelayWorkflow = ".\workflows\autoprompt_relay_ltxv_i2v.json",
    [string]$SinglePromptWorkflow = ".\workflows\autoprompt_ltxv_i2v.json",
    [ValidateSet("auto", "relay", "single_prompt")]
    [string]$RenderMode = "single_prompt",
    [string]$SinglePromptTitle = "#PROMPT",
    [string]$SinglePromptInput = "text",
    [string]$RollingFrameProfile = "original",
    [int]$SmokeScene = 16,
    [switch]$SmokeOnly,
    [switch]$NoSkipExisting,
    [switch]$SkipTests,
    [switch]$SkipRelayCompact,
    [switch]$SkipAnchorFix,
    [switch]$SkipStoryboard,
    [switch]$SkipLtx,
    [switch]$SkipFinalConcat,
    [switch]$DiagnosticOriginalAudioMux,
    [switch]$NoOriginalAudioMux
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-UvPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Script,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & uv run python $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: uv run python $Script"
    }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command"
    }
}

Push-Location $PSScriptRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv not found in PATH."
    }

    $projectConfigPath = (Resolve-Path $ProjectConfig).Path
    $projectConfigDir = Split-Path -Parent $projectConfigPath
    $projectConfigJson = Get-Content -Path $projectConfigPath -Raw | ConvertFrom-Json
    $inputAudio = [string]$projectConfigJson.input_audio
    if (-not [System.IO.Path]::IsPathRooted($inputAudio)) {
        $inputAudio = Join-Path $projectConfigDir $inputAudio
    }

    $songId = [System.IO.Path]::GetFileNameWithoutExtension($inputAudio)
    $projectOutputDir = Join-Path $projectConfigDir "output"
    $timelineDir = Join-Path $projectOutputDir "timeline"
    $promptsDir = Join-Path $projectOutputDir "prompts"
    $renderDir = Join-Path $projectOutputDir "render"

    $stage1Segments = Join-Path $timelineDir "stage1_segments_$songId.json"
    $resolvedContext = Join-Path $promptsDir "resolved_context_$songId.json"
    $conceptPrompts = Join-Path $promptsDir "concept_prompts_$songId.json"
    $sceneDetails = Join-Path $promptsDir "scene_details_$songId.json"
    $scenePrompts = Join-Path $promptsDir "scene_prompts_$songId.json"
    $renderPlan = Join-Path $renderDir "render_plan_$songId.json"
    $compactPlan = Join-Path $renderDir "render_plan_${songId}__compact.json"
    $anchoredPlan = Join-Path $renderDir "render_plan_${songId}__compact_anchored.json"
    $storyboardDir = Join-Path $renderDir "storyboard"
    $ltxDir = Join-Path $renderDir "ltx_$RenderMode"
    if ($SmokeOnly) {
        $ltxDir = Join-Path $renderDir "ltx_${RenderMode}_smoke"
    }
    $ltxDebugDir = Join-Path $renderDir "ltx_${RenderMode}_debug"
    $finalConcatVideo = Join-Path $ltxDir "final_concat_video_only.mp4"
    $finalConcat = Join-Path $ltxDir "final_concat.mp4"
    $finalConcatSceneAudioDebug = Join-Path $ltxDir "final_concat_scene_audio_debug.mp4"
    $concatList = Join-Path $ltxDir "concat_list.txt"

    Write-Host "Project: $projectConfigPath" -ForegroundColor Yellow
    Write-Host "Input audio: $inputAudio" -ForegroundColor Yellow
    Write-Host "Song ID: $songId" -ForegroundColor Yellow
    Write-Host "Render mode: $RenderMode" -ForegroundColor Yellow

    if (-not $SkipTests) {
        Write-Step "Running tests"
        Invoke-UvPython -Script "-m" -Arguments @("unittest", "discover", "-s", "tests")
    }

    Write-Step "Running main pipeline"
    Invoke-UvPython -Script "main.py" -Arguments @(
        "--project", $projectConfigPath,
        "--app-config", $AppConfig,
        "--concept-batch-size", "$ConceptBatchSize"
    )

    $planForNextStep = $renderPlan
    if (-not $SkipRelayCompact -and $RenderMode -ne "single_prompt") {
        Write-Step "Compacting relay prompts"
        Invoke-UvPython -Script "compact_relay_prompts.py" -Arguments @(
            "--app-config", $AppConfig,
            "--input-render-plan", $renderPlan,
            "--output-render-plan", $compactPlan
        )
        $planForNextStep = $compactPlan
    }

    if (-not $SkipAnchorFix) {
        $resolvedContextJson = Get-Content -Path $resolvedContext -Raw | ConvertFrom-Json
        $subjectAnchor = [string]$resolvedContextJson.subject
        if ([string]::IsNullOrWhiteSpace($subjectAnchor)) {
            throw "No subject anchor found in $resolvedContext"
        }

        Write-Step "Fixing prompt anchors"
        Invoke-UvPython -Script "fix_ltx_prompt_anchors.py" -Arguments @(
            "--input-render-plan", $planForNextStep,
            "--output-render-plan", $anchoredPlan,
            "--subject-anchor", $subjectAnchor
        )
        $planForNextStep = $anchoredPlan
    }

    if (-not $SkipStoryboard) {
        Write-Step "Rendering storyboard"
        Invoke-UvPython -Script "render_storyboard.py" -Arguments @(
            "--app-config", $AppConfig,
            "--render-plan", $planForNextStep,
            "--workflow", $StoryboardWorkflow,
            "--output-dir", $storyboardDir
        )
    }

    if (-not $SkipLtx) {
        Write-Step "Rendering LTX"
        $ltxWorkflow = if ($RenderMode -eq "relay") { $RelayWorkflow } else { $SinglePromptWorkflow }
        $ltxArgs = @(
            "--app-config", $AppConfig,
            "--render-plan", $planForNextStep,
            "--workflow", $ltxWorkflow,
            "--audio", $inputAudio,
            "--storyboard-dir", $storyboardDir,
            "--output-dir", $ltxDir,
            "--debug-workflows-dir", $ltxDebugDir,
            "--render-mode", $RenderMode,
            "--rolling-frame-profile", $RollingFrameProfile,
            "--single-prompt-title", $SinglePromptTitle,
            "--single-prompt-input", $SinglePromptInput
        )

        if ($RenderMode -eq "auto") {
            $ltxArgs += @("--single-prompt-workflow", $SinglePromptWorkflow)
        }

        if ($SmokeOnly) {
            $ltxArgs += @("--scenes", "$SmokeScene", "--no-skip-existing")
        }
        elseif ($NoSkipExisting) {
            $ltxArgs += @("--no-skip-existing")
        }

        Invoke-UvPython -Script "render_ltx.py" -Arguments $ltxArgs
    }

    if (-not $SkipFinalConcat -and (Test-Path $concatList)) {
        Write-Step "Final FFmpeg video-only concat"
        Invoke-Native -Command "ffmpeg" -Arguments @(
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", $concatList,
            "-an",
            "-c:v", "copy",
            $finalConcatVideo
        )

        Write-Step "Muxing original full audio"
        Invoke-Native -Command "ffmpeg" -Arguments @(
            "-y",
            "-i", $finalConcatVideo,
            "-i", $inputAudio,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            "-shortest",
            $finalConcat
        )

        if ($DiagnosticOriginalAudioMux) {
            Write-Step "Diagnostic concat with per-scene audio"
            Invoke-Native -Command "ffmpeg" -Arguments @(
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", $concatList,
                "-c", "copy",
                $finalConcatSceneAudioDebug
            )
        }
        elseif ($NoOriginalAudioMux) {
            Write-Host "-NoOriginalAudioMux is deprecated; original-audio muxing is now always used for final_concat.mp4." -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "Pipeline complete." -ForegroundColor Green
    Write-Host "Render plan: $planForNextStep" -ForegroundColor Cyan
    if (Test-Path $finalConcat) {
        Write-Host "Final video: $finalConcat" -ForegroundColor Cyan
    }
    elseif (Test-Path $finalConcatVideo) {
        Write-Host "Video-only concat: $finalConcatVideo" -ForegroundColor Cyan
    }
}
finally {
    Pop-Location
}
