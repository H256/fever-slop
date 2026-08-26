# Strukturierte H3-Prompt-Erzeugung

FeverSlop kann H3-Prompts in zwei Stufen erzeugen:

1. Ein Planner erzeugt ausschließlich strukturierte Sections (`facts`, `shots`,
   `shot_windows` und optionale Referenzbindungen).
2. `DeterministicH3Compiler` sortiert, validiert und formatiert diese Sections zu
   einem stabilen Base- oder Full-Reference-Prompt.

Der strukturierte Pfad wird über `DspyH3PromptBuilder.build_h3_prompt` mit
`structured_sections` aktiviert. Für Batch-Verarbeitung kann
`build_all_h3_prompts` ein Mapping `structured_sections_by_segment` erhalten.
Wenn kein Mapping übergeben wird, bleibt der bestehende DSPy-Renderer aktiv.

## Contract

- Locked facts benötigen `scene_id`, Kategorie, Schlüssel, Wert und `source_id`.
- Jeder Creative-Shot benötigt eine eindeutige `shot_id`, sichtbare Aktion und
  Performance-Beschreibung.
- Für jeden Shot muss ein gültiges Zeitfenster existieren.
- Backend-Labels und Timecodes gehören nicht in kreative Felder; sie werden erst
  beim deterministischen Zusammenbau eingefügt.
- Die Ausgabe enthält `prompt_provenance` mit Compilername und Version, damit
  spätere Render-Ergebnisse reproduzierbar einem Prompt-Compiler zugeordnet
  werden können.

Der Legacy-Pfad wird dadurch nicht automatisch umgestellt. Ein Workflow-Profil
oder Planner kann den strukturierten Pfad gezielt aktivieren, sobald dessen
Sections vollständig vorliegen.
