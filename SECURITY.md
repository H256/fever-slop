# Security policy

## Reporting a vulnerability

Do not disclose credentials, private URLs, exploit details, or personal data in
a public issue. Report security problems privately through GitHub's private
security reporting channel or the contact channel listed in the repository.

Include the affected version or commit, the smallest reliable reproduction,
the impact, and any suggested mitigation. Redact tokens, media, prompts, and
private project data before sending logs.

FeverSlop connects to local and OpenAI-compatible services and can process
user-supplied media. Keep API keys in local ignored configuration or
environment variables and review generated workflows before sending them to
external services.
