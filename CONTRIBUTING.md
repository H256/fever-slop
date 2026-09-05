# Contributing to FeverSlop

Contributions are welcome through the repository's issue tracker and pull requests.

This project has no bug bounty program and does not provide monetary rewards
or guaranteed compensation for bug reports, security reports, or other
contributions. Security issues must follow the private reporting guidance in
`SECURITY.md`, not a public issue.

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

Maintainers may close or remove duplicate, automated, abusive, off-topic, or
spam submissions, including reports that do not contain enough actionable
information to investigate.
