# Contributing to FeverSlop

Contributions are welcome through GitHub issues and pull requests.

## Before opening a pull request

1. Read the relevant documentation under `documentation/`.
2. Keep changes focused and avoid committing generated media, model weights,
   credentials, local configuration, or private project data.
3. Add or update tests for behavior changes.
4. Run the focused tests, then the full checks when practical:

```bash
uv run ruff check .
uv run python -m unittest discover -s tests
```

Use an `agent/` branch prefix unless the change is already being developed on
an existing branch. Explain platform-specific limitations, model requirements,
and external services in the pull request description.

By submitting a contribution, you agree that it may be distributed under the
MIT License used by this project.
