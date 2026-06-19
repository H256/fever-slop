# FeverSlop Music Video Pipeline

FeverSlop erzeugt aus einem Song einen beat- und vocal-synchronen Musikvideo-Renderplan und rendert daraus:

1. Audio-Stems und Vocal-/Lyric-Timeline
2. beat-aligned Szenen mit garantierten Mindest-/Maximaldauern
3. Story-, Konzept-, Z-Image- und LTX-Prompts
4. Z-Image Startframes pro Szene
5. LTX Image-to-Video Clips mit Single-Prompt-I2V
6. eine FFmpeg-Concat-Liste fuer das finale Video

Der aktuelle LTX-Standardpfad ist der Non-Relay-Single-Prompt-Modus wie in den urspruenglichen Workflows. Dafuer nutzt der Renderer pro Szene `ltx.original_style_i2v_prompt`. PromptRelay bleibt optional fuer Workflows mit einem korrekt verdrahteten `#PROMPT_RELAY` Node.

Wenn `render_ltx.py` mit `--project-config` aufgerufen wird, nimmt es `scene_generation.min_duration`, `scene_generation.max_duration` und `lora_1.*` aus dieser Datei, sofern der jeweilige CLI-Wert nicht explizit gesetzt ist. Die Reihenfolge ist: Built-in Default, dann `config.json`, dann Kommandozeile.

Die ausfuehrliche Projektanleitung mit neuer Projektstruktur, Runner-Parametern, allen Config-Keys, CLI-Optionen und Steering-Tutorial liegt hier:

```text
docs/project_workflow.md
```

Schnellstart fuer ein Projekt mit `config.json`:

```powershell
.\test.ps1 .\projects\my_song
```

oder:

```bat
test.bat .\projects\my_song
```

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
â”œâ”€ config.json
â”œâ”€ input/
â”‚  â””â”€ ComfyUI_00056_.mp3
â””â”€ output/
   â”œâ”€ stems/
   â”œâ”€ timeline/
   â”œâ”€ prompts/
   â””â”€ render/
      â”œâ”€ render_plan_ComfyUI_00056_.json
      â”œâ”€ storyboard/
      â””â”€ ltx/
```

Globale App-Konfiguration liegt normalerweise im Repo-Root:

```text
app_config.json
```

Projekt-Konfiguration liegt pro Projekt:

```text
projects/my_frst_project/config.json
```

Nach dem Storyboard-Schritt liegt die statische Review-Seite standardmaessig hier:

```text
projects/my_frst_project/output/render/storyboard/index.html
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
    "base_url": "http://127.0.0.1:8188",
    "model_overrides": []
  }
}
```

Einstellungen:

- `llm.base_url`: OpenAI-kompatible `/v1` API des lokalen oder entfernten LLM-Servers.
- `llm.model`: Modellname, so wie ihn der Server erwartet.
- `llm.temperature`: Kreativitaet der Textgeneration. Fuer reproduzierbarere Prompts eher `0.4` bis `0.7`.
- `llm.max_tokens`: Tokenlimit pro LLM-Antwort. Bei grossen Projekten sind `4096` oft knapp; falls JSON abgeschnitten wird, Batch-Modus verwenden oder erhoehen.
- `comfyui.base_url`: ComfyUI API-Adresse.
- `comfyui.model_overrides`: optionale, strikte Modellnamen-Overrides fuer Sonderfaelle.

Wenn `--app-config` fehlt oder die Datei nicht existiert, nutzt der Code Defaults: `http://localhost:8080/v1` fuer LLM und `http://127.0.0.1:8188` fuer ComfyUI.

ComfyUI-Modellreferenzen in Workflow-JSONs werden vor jedem Renderlauf automatisch gegen den in `app_config.json` konfigurierten Server aufgeloest. Details stehen in:

```text
docs/comfyui_model_resolution.md
```

## Projekt config.json

Minimalbeispiel:

```json
{
  "project_name": "forest_song",
  "input_audio": "input/ComfyUI_00056_.mp3",
  "lyrics": "",
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
- `lyrics`: Optionale vollstaendige Referenzlyrics. Wenn gesetzt, bleiben Timing und Segmentgrenzen aus Whisper/RMS erhalten; nur der erkannte Vocal-Text wird gegen diese Referenz korrigiert.
- `story_idea`: Harte Vorgabe fuer die Story. Wenn leer, generiert der LLM eine Storyidee.
- `style`: Harte Vorgabe fuer den visuellen Stil. Wenn leer, generiert der LLM einen Stilblock.
- `subject`: Harte Vorgabe fuer Hauptfigur/Subjekt. Wichtig fuer konsistente Charaktere.
- `locations`: Liste erlaubter Orte. Nutze konkrete Orte, wenn keine neuen Locations erfunden werden sollen.

### lyrics

Optional complete reference lyrics for the song. When this field is set, FeverSlop still uses Whisper and vocal-energy detection for timing, but corrects the detected vocal segment text against these reference lyrics before building scene prompts.

Use this when Whisper hears the right timing but gets words wrong. Do not use it to force different timing; segment boundaries are preserved.

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

Ausfuehrliche Beispiele zum Unterschied zwischen Top-Level-Feldern, `steering` und `prompt_guidance` stehen in `docs/project_workflow.md` unter "Global Fields vs Steering vs Prompt Guidance".

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

### Storyboard-Review-Seite

Nach dem Rendern der Storyboard-Startframes kann `storyboard_page.py` eine statische HTML-Seite erzeugen:

```powershell
uv run python storyboard_page.py `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-html .\projects\my_frst_project\output\render\storyboard\index.html `
  --title "Storyboard Review"
```

Die Seite ist eine reine Datei-Ausgabe ohne Server. Sie zeigt pro Szene einen Block mit Bild, Szeneninfos, Storybeschreibung und Prompt-Details. Der Modusschalter oben rechts bietet:

- `Full`: Bild plus Story-/Prompt-Kontext.
- `Compact`: echtes Bildgrid; Textbereiche sind ausgeblendet, der Z-Image-Prompt liegt als nativer Browser-Tooltip auf dem anklickbaren Bild.

Das Grid nutzt die volle Seitenbreite, begrenzt sich aber auf maximal fuenf Karten pro Zeile, damit die Bilder lesbar bleiben. Der One-Command-Runner erzeugt `output/render/storyboard/index.html` automatisch nach dem Storyboard-Render; mit `-SkipStoryboardPage` kann dieser Schritt uebersprungen werden.

### LTX PromptRelay-Modus

Das ist ein optionaler Renderpfad fuer Workflows mit PromptRelay-Node.

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
- Mixed-/Wechsel-Szenen koennen optional im PromptRelay-Modus gerendert werden, wenn ein passender Workflow vorhanden ist.

Renderer-Modi:

- `--render-mode single_prompt`: Default, nutzt `--workflow` als Non-Relay-Workflow und patcht `#PROMPT` oder den Fallback `#PROMPT_POSITIVE`.
- `--render-mode relay`: nutzt `--workflow` als PromptRelay-Workflow und patcht `#PROMPT_RELAY`.
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

### 5. Storyboard-Review-Seite erzeugen

```powershell
uv run python storyboard_page.py `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-html .\projects\my_frst_project\output\render\storyboard\index.html
```

Optionen:

| Option | Default | Bedeutung |
| --- | --- | --- |
| `--render-plan` | required | Finaler Renderplan, idealerweise derselbe Plan wie fuer `render_storyboard.py` und `render_ltx.py`. |
| `--storyboard-dir` | required | Ordner mit `scene_XXXX.png`. |
| `--output-html` | `storyboard/index.html` | Ziel der statischen Review-Seite. |
| `--title` | `Storyboard Review` | Seitentitel. |
| `--limit` | none | Nur die ersten N Szenen anzeigen. |
| `--scenes` | none | Szenenauswahl, z. B. `1,2,5-8`. |
| `--allow-missing-images` | off | HTML auch erzeugen, wenn einzelne Bilder fehlen. |

Oeffne danach:

```text
output/render/storyboard/index.html
```

Die Seite hat einen Full-/Compact-Schalter. Compact blendet Ueberschriften und Text aus und zeigt nur die Bilder; der verwendete Bildprompt ist dann als Tooltip am Bildlink verfuegbar.

### 6. Einzelne LTX-Szene testen

Vor einem Vollrender zuerst eine kritische Szene testen, z. B. Szene 16:

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\video_ltxv_i2v_v1.json `
  --render-mode single_prompt `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx_single `
  --scenes 16 `
  --no-skip-existing `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_debug
```

Mit Character-LoRA im LTX-Workflow:

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\video_ltxv_i2v_v1.json `
  --render-mode single_prompt `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx_single `
  --scenes 16 `
  --lora-1-enabled `
  --lora-1-name "characters\my_character.safetensors" `
  --lora-1-strength-model 0.85 `
  --lora-1-strength-clip 0.65 `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_debug
```

Der LTX-Workflow muss den LoRA-Node bereits korrekt in den Model/Clip-Pfad verdrahtet haben. Der Code fuegt keine LoRA-Nodes ein, sondern patcht nur den vorhandenen Node mit `_meta.title` `#LORA_1`. Fuer spaetere Multi-LoRA-Workflows sind `#LORA_2`, `#LORA_3`, ... reserviert.

Workflow-Upgrade:

1. Neuen ComfyUI API-Workflow exportieren.
2. Alle dynamischen Nodes mit stabilen `#...` Titeln versehen.
3. `#LORA_1` in den Model/Clip-Pfad verdrahten.
4. Workflow-Validation laufen lassen.
5. Eine Szene mit `--debug-workflows-dir` rendern und die Debug-JSON pruefen.

Pruefen:

```text
.\projects\my_frst_project\output\render\ltx_single\final\scene_0016.mp4
.\projects\my_frst_project\output\render\ltx_debug\scene_0016_workflow.json
```

Bei aktivem LoRA im Debug-Workflow pruefen, dass `#LORA_1` den erwarteten Dateinamen und die erwarteten Staerken enthaelt.

### 6. Alle LTX-Szenen rendern

Single-Prompt-Workflow:

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\video_ltxv_i2v_v1.json `
  --render-mode single_prompt `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx_single `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_debug
```

PromptRelay-Workflow, falls du einen passenden `#PROMPT_RELAY` Workflow nutzen willst:

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\your_prompt_relay_workflow.json `
  --render-mode relay `
  --audio .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  --storyboard-dir .\projects\my_frst_project\output\render\storyboard `
  --output-dir .\projects\my_frst_project\output\render\ltx_relay `
  --debug-workflows-dir .\projects\my_frst_project\output\render\ltx_relay_debug
```

Auto-Modus mit beiden Workflows:

```powershell
uv run python render_ltx.py `
  --app-config .\app_config.json `
  --render-plan .\projects\my_frst_project\output\render\render_plan_ComfyUI_00056__compact_anchored.json `
  --workflow .\workflows\your_prompt_relay_workflow.json `
  --single-prompt-workflow .\workflows\video_ltxv_i2v_v1.json `
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

Standard ist jetzt ein frame-genauer Video-Concat ohne Szenen-Audio, danach ein einmaliger Mux mit dem kompletten Originalsong. Das vermeidet kleine Audio-Hiccups an Clip-Grenzen.

Video-only Concat:

```powershell
ffmpeg -y -f concat -safe 0 `
  -i .\projects\my_frst_project\output\render\ltx\concat_list.txt `
  -an -c:v copy `
  .\projects\my_frst_project\output\render\ltx\final_concat_video_only.mp4
```

Original-Audio muxen:

```powershell
ffmpeg -y `
  -i .\projects\my_frst_project\output\render\ltx\final_concat_video_only.mp4 `
  -i .\projects\my_frst_project\input\ComfyUI_00056_.mp3 `
  -map 0:v:0 -map 1:a:0 `
  -c:v copy -c:a aac -b:a 320k -shortest `
  .\projects\my_frst_project\output\render\ltx\final_concat.mp4
```

Ein Vergleichsexport mit per-scene Audio ist nur ein Diagnosepfad, z.B. ueber `.\test.ps1 -DiagnosticOriginalAudioMux`.

### 8. Projekt-Assets als ZIP archivieren

Nach einem Renderlauf koennen Arbeits- und Zwischendateien des Projektordners in ein ZIP geschrieben werden:

```powershell
uv run python -m tools.project_asset_archive --project .\projects\my_frst_project\config.json
```

Standardziel:

```text
projects/my_frst_project/archives/my_frst_project_assets_YYYYMMDD_HHMMSS.zip
```

Nicht archiviert werden `config.json`, `output/render/storyboard/**`, finale muxed Videos wie `final_concat.mp4` oder `<project_name>.mp4`, und der `archives/`-Ordner selbst. Bestehende ZIP-Dateien werden nicht ueberschrieben; bei Namenskollisionen wird `-2`, `-3` usw. angehaengt. Der Befehl loescht keine Dateien. Vor dem Schreiben kann die Dateiliste geprueft werden:

```powershell
uv run python -m tools.project_asset_archive --project .\projects\my_frst_project\config.json --dry-run
```

## Safety- und Reparaturbefehle

### Was ist Entry-Point und was ist Bibliothekscode?

Root-Dateien bleiben fuer abwaertskompatible CLI-Aufrufe absichtlich erhalten:

```text
main.py
render_storyboard.py
render_ltx.py
compact_relay_prompts.py
fix_ltx_prompt_anchors.py
storyboard_page.py
normalize_render_plan.py
repair_scene_srt.py
trim_existing_ltx_clips.py
```

Diese Dateien nicht allein deshalb loeschen, weil sie in keinem `import` auftauchen. Neue Implementierung gehoert nach `src/feverslop`.

Die wichtigsten Package-Bereiche:

```text
src/feverslop/
+-- application/    Use-Cases ohne konkrete Adapter
+-- composition/    Verdrahtung von Config, Use-Cases und Adaptern
+-- domain/         Renderplan-, LTX- und Postprocessing-Domain-Typen
+-- ports/          Protocols und Port-Typen
+-- adapters/       ComfyUI, lokale Artefakte, LLM, FFmpeg/Postprocessing
+-- pipeline/       Renderplan-/Timeline-Builder
+-- prompting/      Prompt-Generierung und Prompt-Fixes
+-- tools/          importierbare Tool-Implementierungen
```

Composition Roots:

```text
feverslop.composition.generate_render_plan
feverslop.composition.render_storyboard
```

Der produktive LTX-Renderadapter liegt hier:

```text
feverslop.adapters.comfyui_video_backend
```

Aktuelle Architekturgrenzen:

- `feverslop.application`: Use-Cases und Pipeline-Services ohne konkrete Adapter.
- `feverslop.composition`: verdrahtet Config, Use-Cases und Adapter fuer CLI-Einstiegspunkte.
- `feverslop.domain`: Renderplan-, LTX- und Postprocessing-Domain-Typen.
- `feverslop.ports`: Protocols und Request-Typen; Ports importieren keine Adapter.
- `feverslop.adapters`: ComfyUI, lokale JSON-Artefakte, OpenAI-kompatible LLMs und FFmpeg/Postprocessing.

Die Root-Dateien `ltx_video_renderer.py`, `storyboard_renderer.py` und `workflow_patcher.py` sind Compatibility-Fassaden fuer alte Imports. Details stehen in:

```text
docs/architecture_compatibility.md
```

Wenn Root-Dateien oder alte Tool-Namen entfernt werden sollen, vorher Imports und Dokumentation pruefen:

```powershell
rg -n "DATEINAME_OHNE_PY|python DATEINAME.py|uv run python DATEINAME.py" .
uv run python -m unittest discover -s tests
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
--no-upload-audio --uploaded-audio-name "feverslop/audio/ComfyUI_00056_.mp3"
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
