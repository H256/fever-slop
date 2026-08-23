# Model-neutral subject/action planning

Die Pipeline erzeugt im Prompt-Generierungsschritt pro Szene einen gemeinsamen
Subject-/Action-Plan. Erst danach werden Backend-Prompts komponiert. Einzelne
Subjects werden nicht unabhängig voneinander geplant.

## Ablauf

1. DSPy erhält Szene, Konzept, Scene-Details, Referenzen und globalen Kontext.
2. DSPy erzeugt einen versionierten `subject-directives/v1`-Plan.
3. Der Plan wird im Scene-Prompt-Artefakt unter `subject_directives` gespeichert.
4. H3, LTX T2V, LTX MSR und Ingredients projizieren denselben Plan.
5. Deterministische Prüfungen verwerfen fehlende Subjects, Actions, Props,
   Zeitabdeckung oder widersprüchliche Relationen vor dem Rendering.

Legacy-Szenen ohne `subject_directives` bleiben kompatibel. Bei einer echten
LLM-Konfiguration wird die Erzeugung automatisch ausgeführt; ein fehlerhafter
strukturierter DSPy-Output wird nicht stillschweigend als gültiger Plan übernommen.

## Contract-Beispiel

```json
{
  "schema_version": "subject-directives/v1",
  "shot_id": "scene-50-shot-1",
  "temporal_scope": {"start_seconds": 0, "end_seconds": 4},
  "subjects": [
    {
      "subject_id": "singer",
      "role": "singer",
      "position": "front center",
      "action": "sings into the microphone",
      "prop_bindings": [
        {"prop_id": "microphone", "state": "held"}
      ],
      "visibility": "visible",
      "cardinality": 1,
      "temporal_scope": {"start_seconds": 0, "end_seconds": 4}
    },
    {
      "subject_id": "keyboarder",
      "role": "keyboarder",
      "position": "rear right at the keyboard",
      "action": "plays the keyboard",
      "prop_bindings": [
        {"prop_id": "keyboard", "state": "played"}
      ],
      "visibility": "visible",
      "cardinality": 1,
      "temporal_scope": {"start_seconds": 0, "end_seconds": 4}
    }
  ],
  "spatial_relations": [
    {"subject_id": "singer", "relation": "in_front_of", "target_id": "keyboarder"}
  ]
}
```

`prop_bindings` unterscheidet bewusst zwischen `held`, `played`, `attached`,
`placed` und `absent`. Ein fehlender Prop-Eintrag bedeutet nicht dasselbe wie
`{"state": "absent"}`.

## Python-Projektion

```python
from feverslop.domain.subject_directives import SubjectDirectivePlan
from feverslop.prompting.subject_directive_projections import (
    project_subject_directives,
)

plan = SubjectDirectivePlan.from_dict(payload)
h3 = project_subject_directives(plan, backend="minimax-h3-r2v")
msr = project_subject_directives(plan, backend="ltx-msr")
```

Die Backend-Projektionen dürfen Fakten nur ordnen und formatieren. Die
Coverage-Prüfung schlägt fehl, wenn ein Subject, seine Position oder Aktion,
ein Prop-State oder eine räumliche Relation aus dem resultierenden Prompt
verschwindet.

## Regression-Fälle

Die Fixtures unter `tests/fixtures/subject_directives/regression_scenes.json`
decken ab:

- Szene 47: Rollen- und Instrumentbindung
- Szene 50: keine kollektive Formulierung wie „their instruments“
- Szene 52: explizite Cardinality sowie `absent`-Prop- und Background-Zustände

Die Tests benötigen weder GPU noch ComfyUI oder externe Modellaufrufe.
