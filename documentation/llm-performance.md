# LLM performance

FeverSlop sends prompt-generation requests through the configured
OpenAI-compatible endpoint. Local inference speed depends mostly on model,
context length, generated tokens, concurrency, and server-side reasoning
settings.

## Thinking versus throughput

For short, structured tasks, disabling Thinking on the LLM server can reduce
latency and completion-token usage substantially. This may reduce quality or
coherence for longer creative tasks such as story arcs, character creation,
and scene descriptions. Thinking is therefore a server-side quality/
performance choice; FeverSlop does not switch or validate that mode.

The existing global `llm.model` remains sufficient for a single-model setup.
Optional task-profile model IDs can be added later when different models are
available for creative and structured work.

## Diagnostics

The LLM client records prompt, completion, total, and reported reasoning token
counts, plus model and finish reason. These diagnostics contain no prompt or
completion text. Providers that do not report a field expose it as zero or
unknown; missing reasoning metadata must not be interpreted as proof that no
reasoning occurred.

Keep `llm.max_concurrent_requests` at `1` for local servers where additional
parallel slots reduce per-request generation speed. Increase it only after
measuring the complete pipeline on the target hardware.
