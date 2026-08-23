# Test suites

Tests are assigned explicitly to a suite. `unittest discover -s tests` is not
used by CI because it cannot distinguish unit, integration, and end-to-end
tests.

- `unit`: deterministic tests; no network connections, API keys, external
  services, or media binaries.
- `integration`: multi-layer tests that may use local tools such as FFmpeg.
- `e2e`: production-entry-point tests, run only by manually dispatched CI.

When adding a test module, add it to exactly one suite. Keep fake-port tests in
the unit suite when they do not invoke external processes or services.
