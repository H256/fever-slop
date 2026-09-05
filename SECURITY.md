# Security policy

## Reporting a vulnerability

Do not disclose credentials, private URLs, exploit details, or personal data in
a public issue. Report security problems privately through the repository's
private reporting channel or the contact channel listed in the repository.

FeverSlop does not operate a bug bounty program and does not offer monetary
rewards, gifts, or guaranteed public credit for security reports. There is no
guaranteed response or remediation time. Public vulnerability reports,
duplicates, automated submissions, spam, and reports without actionable
technical information may be closed without further response.

Include the affected version or commit, the smallest reliable reproduction,
the impact, and any suggested mitigation. Redact tokens, media, prompts, and
private project data before sending logs.

FeverSlop connects to local and OpenAI-compatible services and can process
user-supplied media. Keep API keys in local ignored configuration or
environment variables and review generated workflows before sending them to
external services.
