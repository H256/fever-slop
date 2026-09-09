# Canonical music-video benchmark plan

The checked-in fixture at `tests/fixtures/canonical_benchmark/benchmark_plan.json`
is a deterministic, one-minute planning target. It contains ten semantic scenes
with contiguous six-second audio windows, stable scene IDs, explicit reference
expectations, and concrete prompts.

The plan follows Tamsin and Soren through Mirrorline Central. Soren is the only
performer and visible singer; Tamsin never lip-syncs. The red survey thread is a
recurring prop, while the terminal and east platform provide recognizable
locations for continuity checks. The fixture deliberately alternates performance
and non-performance scenes and includes both single-actor and two-actor shots.

This is an authoring and contract fixture, not generated pipeline output. Validate
it with:

```bash
uv run python -m unittest tests.test_canonical_benchmark_fixture
```

The companion continuation fixture at
`tests/fixtures/continuation/continuous_action_24s.json` exercises one
24-second cutless action split into three eight-second technical segments. It
records the absolute 12.0-36.0 second audio interval, the 576 timeline frames,
and the two predecessor-last-frame/start-frame boundaries. Its interruption
fixture renders segment one, fails at segment two, and resumes with segments
two and three.
