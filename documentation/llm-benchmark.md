# Comparing LLM server modes

The benchmark compares the same prompt set against one configured endpoint at
a time. It does not switch Thinking on the server. Start Lemonade/llama.cpp
with one server configuration, run the benchmark, then repeat after changing
the server configuration and compare the JSON output.

```bash
uv run python -m feverslop.tools.llm_benchmark \
  --base-url http://localhost:8080/v1 \
  --model gemma4-26b-a4b \
  --temperature 0.7 \
  --prompt-file documentation/benchmark-prompts.txt \
  > benchmark-no-thinking.json
```

Use a separate output filename such as `benchmark-thinking.json` for the
second run. Compare average latency, completion tokens, reported reasoning
tokens, output word counts, and finish reasons. The report records the
effective `model` and `temperature` settings plus a `completed` flag. If a
request fails, the run stops, `completed` is false, the `error` message is
recorded, and the tool exits non-zero. The benchmark intentionally
does not persist prompt or completion text in its result.
