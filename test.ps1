param(
    [Parameter(Position = 0)]
    [string]$ProjectRoot,
    [string]$ProjectConfig,
    [string]$AppConfig = ".\app_config.json",
    [int]$ConceptBatchSize = 10,
    [string]$StoryboardWorkflow = ".\workflows\image_t2i_startframe_v1.json",
    [string]$RelayWorkflow = "",
    [string]$SinglePromptWorkflow = ".\workflows\video_ltxv_i2v_v1.json",
    [ValidateSet("auto", "relay", "single_prompt")]
    [string]$RenderMode = "single_prompt",
    [string]$SinglePromptTitle = "#PROMPT",
    [string]$SinglePromptInput = "text",
    [string]$RollingFrameProfile = "original",
    [Nullable[double]]$StoryboardLoraStrength,
    [Nullable[double]]$VideoCharacterLoraStrength,
    [Nullable[double]]$VideoLora1StrengthModel,
    [Nullable[double]]$VideoLora1StrengthClip,
    [int]$SmokeScene = 16,
    [switch]$SmokeOnly,
    [switch]$NoSkipExisting,
    [switch]$SkipTests,
    [switch]$SkipMainPipeline,
    [switch]$SkipRelayCompact,
    [switch]$SkipAnchorFix,
    [switch]$SkipStoryboard,
    [switch]$SkipStoryboardPage,
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

function Convert-ToInvariantString {
    param([double]$Value)
    return $Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

function Convert-ToSafeFileStem {
    param(
        [object]$Value,
        [string]$Fallback
    )

    $raw = [string]$Value
    if ([string]::IsNullOrWhiteSpace($raw)) {
        $raw = $Fallback
    }

    $safe = $raw.Trim() -replace "[^A-Za-z0-9._-]+", "_"
    $safe = $safe.Trim("._-".ToCharArray())
    if ([string]::IsNullOrWhiteSpace($safe)) {
        $safe = $Fallback
    }

    return $safe
}

Push-Location $PSScriptRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv not found in PATH."
    }

    if ([string]::IsNullOrWhiteSpace($ProjectConfig)) {
        if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
            $ProjectRoot = ".\projects\my_first_project"
        }

        if (Test-Path -LiteralPath $ProjectRoot -PathType Leaf) {
            $ProjectConfig = $ProjectRoot
            $ProjectRoot = Split-Path -Parent $ProjectConfig
        }
        else {
            $ProjectConfig = Join-Path $ProjectRoot "config.json"
        }
    }

    $projectConfigPath = (Resolve-Path $ProjectConfig).Path
    $projectConfigDir = Split-Path -Parent $projectConfigPath
    $projectConfigJson = Get-Content -Path $projectConfigPath -Raw | ConvertFrom-Json
    $inputAudio = [string]$projectConfigJson.input_audio
    if (-not [System.IO.Path]::IsPathRooted($inputAudio)) {
        $inputAudio = Join-Path $projectConfigDir $inputAudio
    }

    $songId = [System.IO.Path]::GetFileNameWithoutExtension($inputAudio)
    $projectFileStem = Convert-ToSafeFileStem $projectConfigJson.project_name $songId
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
    $storyboardPage = Join-Path $storyboardDir "index.html"
    $ltxDir = Join-Path $renderDir "ltx_$RenderMode"
    if ($SmokeOnly) {
        $ltxDir = Join-Path $renderDir "ltx_${RenderMode}_smoke"
    }
    $ltxDebugDir = Join-Path $renderDir "ltx_${RenderMode}_debug"
    $finalConcatVideo = Join-Path $ltxDir "${projectFileStem}_video_only.mp4"
    $finalConcat = Join-Path $ltxDir "${projectFileStem}.mp4"
    $finalConcatSceneAudioDebug = Join-Path $ltxDir "${projectFileStem}_scene_audio_debug.mp4"
    $concatList = Join-Path $ltxDir "concat_list.txt"

    Write-Host "Project: $projectConfigPath" -ForegroundColor Yellow
    Write-Host "Input audio: $inputAudio" -ForegroundColor Yellow
    Write-Host "Song ID: $songId" -ForegroundColor Yellow
    Write-Host "Render mode: $RenderMode" -ForegroundColor Yellow

    if (-not $SkipTests) {
        Write-Step "Running tests"
        Invoke-UvPython -Script "-m" -Arguments @("unittest", "discover", "-s", "tests")
    }

    if (-not $SkipMainPipeline) {
        Write-Step "Running main pipeline"
        Invoke-UvPython -Script "main.py" -Arguments @(
            "--project", $projectConfigPath,
            "--app-config", $AppConfig,
            "--concept-batch-size", "$ConceptBatchSize"
        )
    }
    else {
        Write-Host "Skipping main pipeline; using existing timeline, prompts, and render plan." -ForegroundColor Yellow
    }

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
        $storyboardArgs = @(
            "--app-config", $AppConfig,
            "--render-plan", $planForNextStep,
            "--workflow", $StoryboardWorkflow,
            "--output-dir", $storyboardDir
        )

        if ($null -ne $StoryboardLoraStrength) {
            $storyboardArgs += @("--character-lora-strength", (Convert-ToInvariantString $StoryboardLoraStrength))
        }

        Invoke-UvPython -Script "render_storyboard.py" -Arguments $storyboardArgs
    }

    if (-not $SkipStoryboardPage) {
        Write-Step "Generating storyboard page"
        Invoke-UvPython -Script "storyboard_page.py" -Arguments @(
            "--render-plan", $planForNextStep,
            "--storyboard-dir", $storyboardDir,
            "--output-html", $storyboardPage
        )
    }

    if (-not $SkipLtx) {
        Write-Step "Rendering LTX"
        if ($RenderMode -ne "single_prompt" -and [string]::IsNullOrWhiteSpace($RelayWorkflow)) {
            throw "RenderMode '$RenderMode' requires -RelayWorkflow pointing to a workflow with #PROMPT_RELAY."
        }

        $ltxWorkflow = if ($RenderMode -eq "single_prompt") { $SinglePromptWorkflow } else { $RelayWorkflow }
        $ltxArgs = @(
            "--app-config", $AppConfig,
            "--project-config", $projectConfigPath,
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

        if ($null -ne $VideoCharacterLoraStrength) {
            $ltxArgs += @("--character-lora-strength", (Convert-ToInvariantString $VideoCharacterLoraStrength))
        }

        if ($null -ne $VideoLora1StrengthModel) {
            $ltxArgs += @("--lora-1-strength-model", (Convert-ToInvariantString $VideoLora1StrengthModel))
        }

        if ($null -ne $VideoLora1StrengthClip) {
            $ltxArgs += @("--lora-1-strength-clip", (Convert-ToInvariantString $VideoLora1StrengthClip))
        }

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
