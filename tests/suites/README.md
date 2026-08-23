# Test suites

Tests are assigned explicitly to a suite. `unittest discover -s tests` is not
used by CI because it cannot distinguish unit, integration, and end-to-end
tests.

- `unit`: deterministic tests; no network connections, API keys, external
  services, or media binaries.
- `integration`: multi-layer tests that may use local tools such as FFmpeg;
  run locally with `uv run python -m tests.suites.integration`.
- `e2e`: production-entry-point tests; run locally with
  `uv run python -m tests.suites.e2e`.

CI intentionally runs neither integration nor E2E tests. They may require
local services, binaries, models, or project data and must never start those
dependencies on a CI runner.

When adding a test module, add it to exactly one suite. Keep fake-port tests in
the unit suite when they do not invoke external processes or services.
