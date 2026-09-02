# Model-neutral subject/action planning

During prompt generation, the pipeline creates a shared subject/action plan for
each scene. Backend prompts are composed only afterward. Individual subjects
are not planned independently from one another.

## Process

1. DSPy receives the scene, concept, scene details, references, and global context.
2. DSPy produces a versioned `subject-directives/v1` plan.
3. The plan is stored under `subject_directives` in the scene prompt artifact.
4. H3, LTX T2V, LTX MSR, and Ingredients project the same plan.
5. Deterministic checks reject missing subjects, actions, or props; incomplete
   time coverage; and contradictory relations before rendering.

Legacy scenes without `subject_directives` remain compatible. With a real LLM
configuration, generation runs automatically; malformed structured DSPy output
is not silently accepted as a valid plan.

## Contract example

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

`prop_bindings` deliberately distinguishes between `held`, `played`, `attached`,
`placed`, and `absent`. A missing prop entry does not mean the same thing as
`{"state": "absent"}`.

## Python projection

```python
from feverslop.domain.subject_directives import SubjectDirectivePlan
from feverslop.prompting.subject_directive_projections import (
    project_subject_directives,
)

plan = SubjectDirectivePlan.from_dict(payload)
h3 = project_subject_directives(plan, backend="minimax-h3-r2v")
msr = project_subject_directives(plan, backend="ltx-msr")
```

Backend projections may only order and format facts. The coverage check fails
if a subject, its position or action, a prop state, or a spatial relation
disappears from the resulting prompt.

## Regression cases

The fixtures in `tests/fixtures/subject_directives/regression_scenes.json`
cover:

- Scene 47: role and instrument binding
- Scene 50: no collective wording such as "their instruments"
- Scene 52: explicit cardinality plus `absent` prop and background states

The tests require neither a GPU nor ComfyUI or external model calls.
