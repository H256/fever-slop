# Autoprompter Music Video Pipeline

Autoprompter erzeugt aus einem Song einen beat- und vocal-synchronen Musikvideo-Renderplan und rendert daraus:

1. Audio-Stems und Vocal-/Lyric-Timeline
2. beat-aligned Szenen mit garantierten Mindest-/Maximaldauern
3. Story-, Konzept-, Z-Image- und LTX-Prompts
4. Z-Image Startframes pro Szene
5. LTX Image-to-Video Clips mit PromptRelay-Steuerung
6. eine FFmpeg-Concat-Liste fuer das finale Video

Der aktuelle LTX-Standardpfad ist weiterhin der segmentierte PromptRelay-Modus. Der Renderer kann aber auch einen Non-Relay-Workflow mit einem einzelnen Prompt-Node rendern. Dafuer nutzt er pro Szene `ltx.original_style_i2v_prompt`; im Auto-Modus entscheidet `ltx.render_mode_hint`.

## Voraussetzungen

- Python 3.12
- `uv`
- FFmpeg im `PATH`
- ComfyUI mit passenden API-Workflows
- ein OpenAI-kompatibler LLM-Endpunkt fuer Textgenerierung
- optional CUDA/PyTorch fuer Demucs/Whisper, je nach lokaler Installation

Installieren:

```powershell
uv sync
```

## Dateien

Typische Projektstruktur:

```text
projects/my_frst_project/
├─ config.json
├─ input/
│  └─ ComfyUI_00056_.mp3
└─ output/
   ├─ stems/
   ├─ timeline/
   ├─ prompts/
   └─ render/
      ├─ render_plan_ComfyUI_00056_.json
      ├─ storyboard/
      └─ ltx/
```

Globale App-Konfiguration liegt normalerweise im Repo-Root:

```text
app_config.json
```

Projekt-Konfiguration liegt pro Projekt:

```text
projects/my_frst_project/config.json
```

## app_config.json

`app_config.json` beschreibt Infrastruktur, nicht den Song.

```json
{
  "llm": {
    "base_url": "http://llm.elysium.lan/v1",
    "model": "gemma4-26b-a4b:instruct",
    "temperature": 0.7,
    "max_tokens": 4096
  },
  "comfyui": {
    "base_url": "http://127.0.0.1:8188"
  }
}
```

Einstellungen:

- `llm.base_url`: OpenAI-kompatible `/v1` API des lokalen oder entfernten LLM-Servers.
- `llm.model`: Modellname, so wie ihn der Server erwartet.
- `llm.temperature`: Kreativitaet der Textgeneration. Fuer reproduzierbarere Prompts eher `0.4` bis `0.7`.
- `llm.max_tokens`: Tokenlimit pro LLM-Antwort. Bei grossen Projekten sind `4096` oft knapp; falls JSON abgeschnitten wird, Batch-Modus verwenden oder erhoehen.
- `comfyui.base_url`: ComfyUI API-Adresse.

Wenn `--app-config` fehlt oder die Datei nicht existiert, nutzt der Code Defaults: `http://localhost:8080/v1` fuer LLM und `http://127.0.0.1:8188` fuer ComfyUI.

## Projekt config.json

Minimalbeispiel:

```json
{
  "project_name": "forest_song",
  "input_audio": "input/ComfyUI_00056_.mp3",
  "video": {
    "fps": 24,
    "width": 1280,
    "height": 704
  },
  "audio": {
    "demucs_model": "htdemucs_ft",
    "whisper_model": "large",
    "language": "de"
  },
  "scene_generation": {
    "min_duration": 2.0,
    "max_duration": 10.0,
    "bias": 0.7,
    "duration_preset": "impact_weighted",
    "seed": 42
  },
  "vocal_detection": {
    "merge_gap": 0.5,
    "min_vocal_duration": 0.4,
    "min_silence_duration": 0.8,
    "rms_low_percentile": 20,
    "rms_high_percentile": 85,
    "rms_ratio": 0.35,
    "smooth_frames": 10
  },
  "story_idea": "",
  "style": "",
  "subject": "",
  "locations": [],
  "steering": {
    "global": "",
    "story_idea": "",
    "style": "",
    "subject": "",
    "locations": "",
    "concepts": "",
    "zimage": "",
    "ltx": "",
    "final_prompts": ""
  },
  "prompt_guidance": {
    "character_visibility": "",
    "shot_types": "",
    "environments": "",
    "lighting": "",
    "camera_motion": "",
    "physical_interaction": "",
    "facial_expression": "",
    "outfit_rules": "",
    "prompt_structure": "",
    "list_handling": "",
    "word_count_min": 40,
    "word_count_max": 50
  }
}
```

### Top-Level

- `project_name`: Anzeigename und Fallback-ID.
- `input_audio`: Pfad zur Audiodatei, relativ zur `config.json` oder absolut.
- `story_idea`: Harte Vorgabe fuer die Story. Wenn leer, generiert der LLM eine Storyidee.
- `style`: Harte Vorgabe fuer den visuellen Stil. Wenn leer, generiert der LLM einen Stilblock.
- `subject`: Harte Vorgabe fuer Hauptfigur/Subjekt. Wichtig fuer konsistente Charaktere.
- `locations`: Liste erlaubter Orte. Nutze konkrete Orte, wenn keine neuen Locations erfunden werden sollen.

### video

- `fps`: Ziel-Framerate. Aktuell meist `24`.
- `width`, `height`: Zielaufloesung fuer Storyboard und LTX. Muss zu den ComfyUI-Workflows passen.

### audio

- `demucs_model`: Stem-Separation-Modell. Default `htdemucs_ft`.
- `whisper_model`: Whisper-Modell fuer Transkription. Default `large`.
- `language`: Sprachcode fuer Whisper, z. B. `de` oder `en`.

### scene_generation

Diese Werte bestimmen die Szenenfenster vor allen Prompt- und Render-Schritten.

- `min_duration`: Mindestdauer pro Szene in Sekunden. Zu kurze Szenen werden gemerged.
- `max_duration`: Maximaldauer pro Szene in Sekunden. Zu lange Szenen werden gesplittet.
- `bias`: Gewichtung der Beat-/Impact-Auswahl. Hoeher bedeutet staerkere Orientierung an Impact-Punkten.
- `duration_preset`: Preset fuer Szenenlaengenlogik. Aktuell ueblich: `impact_weighted`.
- `seed`: Seed fuer reproduzierbare Szenenaufteilung.

Wichtige Regel:

```text
scene_generation.min_duration <= jede Szene <= scene_generation.max_duration
```

Die Pipeline erzeugt zuerst eine rohe SRT und repariert sie danach. Alle folgenden Schritte verwenden die reparierte SRT.

### vocal_detection

Diese Werte bestimmen, wann Audio als Vocal oder instrumental gilt.

- `merge_gap`: Kleine Luecken zwischen gleichartigen Segmenten werden gemerged.
- `min_vocal_duration`: Vocals kuerzer als dieser Wert werden verworfen oder absorbiert.
- `min_silence_duration`: Sehr kurze Pausen werden nicht als stabiler Silent-Abschnitt behandelt.
- `rms_low_percentile`, `rms_high_percentile`: Pegel-Perzentile fuer RMS-Schwellen.
- `rms_ratio`: Pegelverhaeltnis fuer Vocal-Aktivitaet.
- `smooth_frames`: Glaettung der Vocal-Erkennung.

Wenn zu viele falsche Singing-Abschnitte entstehen, `min_vocal_duration` und `min_silence_duration` erhoehen. Wenn echte kurze Vocals fehlen, beide vorsichtig senken.

### steering

`steering` sind Zusatzanweisungen. Sie ersetzen nicht zwingend die Top-Level-Werte, sondern lenken einzelne LLM-Schritte.

- `global`: Wird in mehrere globale Kontextschritte aufgenommen.
- `story_idea`: Zusatzlenkung fuer Story-Generierung.
- `style`: Zusatzlenkung fuer Stil-Generierung.
- `subject`: Zusatzlenkung fuer Subjekt-/Charakter-Generierung.
- `locations`: Zusatzlenkung fuer erlaubte Locations.
- `concepts`: Zusatzlenkung fuer Konzept-Prompts pro Segment.
- `zimage`: Zusatzlenkung fuer Z-Image Startframe-Prompts.
- `ltx`: Zusatzlenkung fuer LTX-Bewegungs-/I2V-Prompts.
- `final_prompts`: reserviert fuer finale Prompt-Stufe.

Pragmatische Empfehlung: Wenn etwas wirklich fest sein muss, nutze Top-Level `story_idea`, `style`, `subject`, `locations`. Nutze `steering.*` fuer weiche Hinweise wie "mehr Close-ups", "keine neuen Charaktere", "Kamera ruhig halten".

### prompt_guidance

`prompt_guidance` ist die konsolenseitige Entsprechung zu UI-Listen fuer die LLM-Prompt-Erzeugung. Die Werte werden in die Konzept-, Detail-, Z-Image- und I2V-Prompt-Calls gegeben. Sie sind Leitplanken, keine Renderparameter.

- `character_visibility`: Vorgaben wie "subject always visible", "medium close-up", "full body".
- `shot_types`: Shot-Liste, z. B. "close-up, medium shot, wide establishing shot".
- `environments`: erlaubte oder bevorzugte Umgebungen innerhalb der Top-Level-`locations`.
- `lighting`: Lichtvorgaben, z. B. "soft rim light, flickering practical lights".
- `camera_motion`: Bewegungsoptionen fuer Kamera, z. B. "slow push-in, handheld orbit".
- `physical_interaction`: sichtbare Aktionen ohne neue Story-Events.
- `facial_expression`: Ausdrucks-/Emotion-Liste.
- `outfit_rules`: harte Regeln fuer Kleidung und Kontinuitaet.
- `prompt_structure`: optionale Strukturvorgabe fuer Concept-Prompts.
- `list_handling`: z. B. "cycle", "random", "reference only" als Hinweis fuer Variation.
- `word_count_min`, `word_count_max`: Zielbereich fuer Concept-Prompts.

Wichtig: Segmenttyp und Vocal-Erkennung haben Vorrang. Bei `instrumental` werden keine Singing- oder Lip-Sync-Begriffe erzwungen, auch wenn eine Guidance-Liste Performance-Begriffe enthaelt.

## Modi

### Konzept-Batch-Modus

Aktiviert ueber `main.py --concept-batch-size N`.

- `0`: Batch-Modus aus, alle Konzepte in einem LLM-Call.
- `5` bis `10`: empfohlen fuer laengere Songs oder Modelle, die JSON am Ende abschneiden.

Der Batch-Modus betrifft nur:

```text
stage1_segments.json -> concept_prompts.json
```

Alles danach bleibt gleich.

### Storyboard-Modus

Es gibt zwei Wege:

1. Direkt mit `main.py --render-storyboard --zimage-workflow ...`
2. Separat mit `render_storyboard.py`

Empfohlen ist der separate Weg, weil du vorher Renderplan-Fixes wie Compact/Anchor anwenden kannst.

### LTX PromptRelay-Modus

Das ist aktuell der aktive Renderpfad.

- `ltx.base_prompt`: globale Szenenbeschreibung.
- `ltx.prompt_relay`: framebasierte lokale Steering-Segmente fuer Singing/Silent-Wechsel.
- `#PROMPT_RELAY.segment_lengths`: wird aus Szenenframes, Preroll und Tail berechnet.

Kurze Relay-Segmente unter 6 Frames werden in Nachbarsegmente gemerged. Dadurch entstehen keine `...,1,...` Relay-Laengen mehr.

### Original-Style Single-Prompt-Modus

Der Renderplan enthaelt nun pro Szene:

```json
"ltx": {
  "original_style_i2v_prompt": "...",
  "render_mode_hint": "single_prompt"
}
```

Dieser Modus ist ueber `render_ltx.py --render-mode single_prompt` nutzbar. Er erwartet einen Workflow ohne PromptRelay, bei dem ein einzelner Prompt-Node gepatcht wird.

Defaults fuer den Single-Prompt-Node:

```text
Node-Titel: #PROMPT
Input-Name: text
```

Wenn dein Workflow anders benannt ist, z. B. der alte Repo-Workflow mit `#PROMPT_POSITIVE`, nutze:

```powershell
--single-prompt-title "#PROMPT_POSITIVE" --single-prompt-input "text"
```

`render_mode_hint`:

- `single_prompt`: Szene ist durchgehend vocal oder durchgehend instrumental.
- `relay`: Szene hat interne Singing/Silent-Wechsel oder ist `mixed`.

Policy:

- Vocal-only Szenen duerfen Singing/Lip-Sync enthalten.
- Instrumental-only Szenen enthalten keine Singing-/Lip-Sync-Begriffe.
- Mixed-/Wechsel-Szenen bleiben zunaechst im PromptRelay-Modus.

Renderer-Modi:

- `--render-mode relay`: nutzt `--workflow` als PromptRelay-Workflow und patcht `#PROMPT_RELAY`.
- `--render-mode single_prompt`: nutzt `--workflow` oder `--single-prompt-workflow` als Non-Relay-Workflow und patcht `#PROMPT`.
- `--render-mode auto`: nutzt pro Szene `ltx.render_mode_hint`; dafuer muessen `--workflow` und `--single-prompt-workflow` gesetzt sein.

### Rolling-Frames-Modus

Aktiv ueber `render_ltx.py --preroll-frames` und `--tail-loss-frames`.

Profile:

- `--rolling-frame-profile original`: Default, `pre=50`, `tail=25`, mit `8N+1`-Rundung.
- `--rolling-frame-profile safe`: Low-VRAM-Profil, `pre=6`, `tail=0`, keine `8N+1`-Rundung.
- `--rolling-frame-profile off`: `pre=0`, `tail=0`.

Originalwerte aus Node 287:

```text
tail_loss_frames = 25
pre_frames       = 50
```

Wichtig: Die Widget-Reihenfolge in der Node ist `tail_loss_frames` dann `pre_frames`. Die im Workflow stehenden letzten Werte `25, 50` bedeuten codeseitig also `tail=25`, `pre/preroll=50`.

Die Original-Node rechnet:

```text
base_frames_for_ltx = truth_frames + PRE_FRAMES + TAIL_FRAMES
frames_for_ltx      = round_up_8n1(base_frames_for_ltx)
```

Das bedeutet: `original` rendert effektiv `50 + 25 = 75` Zusatzframes plus bis zu 7 Padding-Frames fuer `8N+1`. Bei 25 fps sind das etwa 3 Sekunden mehr LTX-Latent pro Szene. Auf kleinen GPUs kann das OOM verursachen.

Default:

```powershell
--rolling-frame-profile original
```

Low-VRAM:

```powershell
--rolling-frame-profile safe
```

Manuelle Overrides:

```powershell
--preroll-frames 6 --tail-loss-frames 6
```

Wirkung:

- LTX rendert einige Frames vor der eigentlichen Szene mit.
- Das finale Clip wird danach getrimmt.
- Die Audiospur bleibt pro Szenenclip das passende, ebenfalls getrimmte Audiofenster.
- Das reduziert sichtbare Luecken zwischen einzeln gerenderten Clips.

Output:

```text
output/render/ltx/
+-- raw/
|   +-- scene_0001_raw.mp4
+-- final/
|   +-- scene_0001.mp4
+-- concat_list.txt
+-- render_manifest.json
```

### PromptRelay Frame-Regel

Der vorhandene Workflow erwartet standardmaessig:

```text
sum(segment_lengths) = #FRAMES - 1
```

Deshalb ist der Default:

```powershell
--segment-length-mode frames_minus_one
```

Falls ein anderer PromptRelay-Node `sum(segment_lengths) = #FRAMES` erwartet:

```powershell
--segment-length-mode frames
```

## Empfohlene Befehlsreihenfolge

Die Beispiele nutzen:

```powershell
$Project = ".\projects\my_frst_project"
$Song = "ComfyUI_00056_"
$RenderPlan = "$Project\output\render\render_plan_$Song.json"
```

PowerShell-Variablen sind optional; du kannst die Pfade auch direkt schreiben.

### 1. Hauptpipeline ausfuehren

```powershell
uv run python main.py `
  --project .\projects\my_frst_project\config.json `
  --app-config .\app_config.json `
  --concept-batch-size 10
```

Erzeugt unter anderem:

```text
output/stems/
output/timeline/timeline_<song>.json
output/timeline/scenes_<song>_raw.srt
output/timeline/scenes_<song>.srt
output/timeline/stage1_segments_<song>.json
output/prompts/ltx_prompt_relay_<song>.json
output/prompts/resolved_context_<song>.json
output/prompts/concept_prompts_<song>.json
output/prompts/scene_details_<song>.json
output/prompts/scene_prompts_<song>.json
output/render/render_plan_<song>.json
```

### 2. Optional: Relay-Prompts kompaktieren

Empfohlen, wenn LTX zu stark driftet oder Relay-Prompts zu lang sind.

```powershell
uv run python compact_relay_prompts.py `
  --app-config .\app_config.json `
  --input-render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056_.json `
  --output-render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact.json `
  --max-words 28
```

Ab hier im Zweifel mit `render_plan_ComfyUI_00056__compact.json` weiterarbeiten.

### 3. Optional: LTX Anchors fixen

Empfohlen, wenn LTX vom Startframe wegdriftet, z. B. nur Baum, Rinde, Seil, Schatten oder Makrodetails zeigt.

```powershell
uv run python fix_ltx_prompt_anchors.py `
  --input-render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact.json `
  --output-render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --subject-anchor "the old weary warrior man with weathered scarred face, salt-and-pepper beard, tattered leather armor, and heavy frayed cloak"
```

Ab hier im Zweifel mit `render_plan_ComfyUI_00056__compact_anchored.json` weiterarbeiten.

### 4. Storyboard Startframes rendern

```powershell
uv run python render_storyboard.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\zimage_api.json `
  --output-dir .\projects\my_frst_project\output\render\storyboard `
  --no-skip-existing
```

Wenn dein Workflow andere Node-Titel nutzt:

```powershell
--positive-title "#PROMPT_POSITIVE" `
--negative-title "#PROMPT_NEGATIVE" `
--save-title "#SAVE_IMAGE" `
--character-lora-title "#CHARACTER_LORA"
```

### 5. Einzelne LTX-Szene testen

Vor einem Vollrender zuerst eine kritische Szene testen, z. B. Szene 16:

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\autoprompt_relay_ltxv_i2v.json `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx `
  --scenes 16 `
  --no-skip-existing `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_debug
```

Pruefen:

```text
.\projects\my_frst_project\output\render\ltx\final\scene_0016.mp4
.\projects\my_frst_project\output\render\ltx_debug\scene_0016_workflow.json
```

### 6. Alle LTX-Szenen rendern

PromptRelay-Workflow:

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\autoprompt_relay_ltxv_i2v.json `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_debug
```

Non-Relay Single-Prompt-Workflow:

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\autoprompt_ltxv_i2v.json `
  --render-mode single_prompt `
  --single-prompt-title "#PROMPT" `
  --single-prompt-input "text" `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx_single `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_single_debug
```

Auto-Modus mit beiden Workflows:

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\autoprompt_relay_ltxv_i2v.json `
  --single-prompt-workflow .\workflows\autoprompt_ltxv_i2v.json `
  --render-mode auto `
  --single-prompt-title "#PROMPT" `
  --single-prompt-input "text" `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx_auto `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_auto_debug
```

Wenn dein Non-Relay-Workflow den vorhandenen Titel `#PROMPT_POSITIVE` nutzt:

```powershell
--single-prompt-title "#PROMPT_POSITIVE" --single-prompt-input "text"
```

Nur bestimmte Szenen:

```powershell
--scenes 1,2,5-8
```

Nur die ersten N Szenen:

```powershell
--limit 5
```

Bestehende Clips neu rendern:

```powershell
--no-skip-existing
```

### 7. Finales Video concatieren

Standard ist Original-Parity: Die finalen Szenenclips werden mit ihren jeweiligen Szenen-Audiospuren concateniert. Der komplette Originalsong wird nicht nachtraeglich ueber das finale Video gemuxt.

Streamcopy:

```powershell
ffmpeg -f concat -safe 0 `
  -i .\projects\my_frst_project\output\render\ltx\concat_list.txt `
  -c copy `
  .\projects\my_frst_project\output\render\ltx\final_video.mp4
```

Falls Streamcopy wegen Codec-/Containerdetails scheitert, re-encoden:

```powershell
ffmpeg -f concat -safe 0 `
  -i .\projects\my_frst_project\output\render\ltx\concat_list.txt `
  -c:v libx264 -crf 18 -preset slow `
  -c:a aac -b:a 192k `
  .\projects\my_frst_project\output\render\ltx\final_video_reencoded.mp4
```

Ein Vergleichsexport mit komplettem Originalsong ist nur ein Diagnosepfad, z.B. ueber `.\test.ps1 -DiagnosticOriginalAudioMux`.

## Safety- und Reparaturbefehle

### Was ist Entry-Point und was ist Bibliothekscode?

Einige Dateien werden nirgends importiert, sind aber trotzdem absichtlich vorhanden, weil sie direkt per CLI aufgerufen werden:

```text
main.py
render_storyboard.py
render_ltx.py
compact_relay_prompts.py
fix_ltx_prompt_anchors.py
```

Diese Dateien nicht allein deshalb loeschen, weil sie in keinem `import` auftauchen.

Wartungs- und Reparatur-Tools liegen unter:

```text
tools/
+-- normalize_render_plan.py
+-- repair_scene_srt.py
+-- trim_existing_ltx_clips.py
+-- render_plan_normalizer.py
```

Die gleichnamigen Root-Dateien sind nur Compatibility-Wrappers fuer alte Befehle.

Aktuelle statische Dead-Code-Kandidaten:

- `prompt_pipeline_batch_patch.py`: wirkt wie ein alter manueller Patch-Referenzstand. Die Batch-Logik ist inzwischen in `concept_prompt_batcher.py` und `main.py` integriert.
- `extract_lyrics.py`: wird vom Hauptpfad nicht verwendet. Es importiert `noise_reduction.py`; beide koennen Legacy-/Experiment-Code sein, solange du sie nicht separat nutzt.

Sicher loeschbar ist nur, was du nicht als manuelles Tool behalten willst. Vor dem Loeschen:

```powershell
rg -n "DATEINAME_OHNE_PY|python DATEINAME.py|uv run python DATEINAME.py" .
```

Danach Tests ausfuehren:

```powershell
uv run python -m unittest discover -s tests
uv run python -m compileall .
```

### Renderplan-Dauern normalisieren

Nur verwenden, wenn ein existierender Renderplan falsche Dauern enthaelt. Besser ist normalerweise ein frischer Lauf von `main.py`.

```powershell
uv run python -m tools.normalize_render_plan `
  --input-render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056_.json `
  --output-render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__duration_fixed.json `
  --min-duration 2.0 `
  --max-duration 10.0
```

### LTX ohne Uploads rendern

Nuetzlich, wenn Audio oder Startframes bereits in ComfyUI liegen:

```powershell
--no-upload-audio --uploaded-audio-name "autoprompter/audio/ComfyUI_00056_.mp3"
```

```powershell
--no-upload-startframes
```

### Seeds

Deterministisch pro Szene:

```powershell
--seed-offset 100000
```

Zufaellig pro Render:

```powershell
--randomize-seed
```

## Debug-Checkliste

Wenn ein Clip 0 Sekunden lang ist oder PromptRelay Fehler wirft, die Debug-Workflow-Datei pruefen:

```text
#FRAMES
#FRAMERATE
#TRIM_AUDIO.start_index
#TRIM_AUDIO.duration
#PROMPT_RELAY.global_prompt
#PROMPT_RELAY.local_prompts
#PROMPT_RELAY.segment_lengths
```

Erwartung bei Default-Modus:

```text
sum(segment_lengths) = #FRAMES - 1
keine segment_lengths kleiner als 6, ausser es gibt nur ein einziges Segment
```

Bei 24 fps und `min_duration = 2.0`:

```text
#FRAMES >= 49
#TRIM_AUDIO.duration ungefaehr >= 2.0
```

Wenn LTX vom Startframe wegdriftet:

1. Compact Relay ausfuehren.
2. Anchor-Fix ausfuehren.
3. `ltx.base_prompt` und `ltx.prompt_relay[].prompt` im Renderplan pruefen.
4. Storyboard mit demselben finalen Renderplan neu rendern.

Wenn Vocals falsch erkannt werden:

1. `output/timeline/timeline_<song>.json` pruefen.
2. `vocal_detection` in `config.json` anpassen.
3. `main.py` neu ausfuehren.

Wenn Szenen zu kurz oder zu lang sind:

1. `scene_generation.min_duration` und `max_duration` pruefen.
2. `output/timeline/scenes_<song>.srt` pruefen.
3. Nicht erst beim LTX-Render reparieren, sondern Renderplan frisch erzeugen.

## Aktuelle Einschraenkungen

- `single_prompt` ist verdrahtet, aber nur so gut wie der uebergebene Non-Relay-Workflow und dessen Prompt-Node-Konvention.
- `auto` braucht zwei Workflows: `--workflow` fuer Relay und `--single-prompt-workflow` fuer Single-Prompt-Szenen.
- Rolling Frames reduzieren Clip-Uebergaenge, beheben aber keine falschen oder driftenden Prompts.
- Storyboard und LTX muessen mit demselben finalen Renderplan gerendert werden, sonst koennen Startframe und I2V-Prompt auseinanderlaufen.
