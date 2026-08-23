# LLM Token Limits

The `max_tokens` value limits the complete generated response. It is not a
per-item limit inside a JSON response.

## Concept batches

Multi-item structured generation reserves a per-item output budget plus a
1024-token JSON overhead reserve. The general formula is:

```text
max_tokens = per_item_limit * item_count + json_overhead
```

Concept generation uses 512 tokens per scene. Lyrics alignment uses 256 tokens
per vocal segment. MSR relay generation uses 512 tokens per relay.

Examples:

| Batch size | Output limit |
|---:|---:|
| 1 | 1,536 |
| 5 | 3,584 |
| 10 | 6,144 |

For 13 lyric segments the limit is 4,352 tokens. For 4 MSR relays it is
3,072 tokens.

Repair calls and other structured tasks retain their task-specific limits.
The global `llm.max_tokens` setting is the DSPy LM default; it does not replace
these per-request limits.
