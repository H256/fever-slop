"""
Patch instructions for main.py.

1. Add import:

from concept_prompt_batcher import ConceptPromptBatcher

2. Add argparse parameter:

parser.add_argument(
    "--concept-batch-size",
    type=int,
    default=0,
    help="Generate concept prompts in batches of N segments. 0 disables batching.",
)

3. Replace the concept prompt generation block with the block below.
"""

# ------------------------------------------------------------------
log_step("8. Concept Prompts")

concept_batch_size = int(getattr(args, "concept_batch_size", 0) or 0)

if concept_batch_size > 0:
    console.print(
        f"[cyan]Using batched concept generation: "
        f"{concept_batch_size} segments per batch[/cyan]"
    )

    concept_batcher = ConceptPromptBatcher(
        llm=llm,
        batch_size=concept_batch_size,
    )

    concept_prompts = run_spinner(
        f"Generating concept prompts in batches of {concept_batch_size}...",
        lambda: concept_batcher.create_concept_prompts_batched(
            stage1_segments=stage1_segments,
            story_idea=concept_story_input,
            global_context=global_context,
            notes=get_steering_value(config, "concepts"),
        ),
    )
else:
    concept_prompts = run_spinner(
        "Generating concept prompts for all scenes...",
        lambda: call_with_supported_kwargs(
            prompt_pipeline.create_concept_prompts,
            stage1_segments=stage1_segments,
            story_idea=concept_story_input,
            global_context=global_context,
            notes=get_steering_value(config, "concepts"),
        ),
    )

# Validate exact coverage before continuing.
missing_concepts = [
    seg["segment_id"]
    for seg in stage1_segments
    if seg["segment_id"] not in concept_prompts
]

extra_concepts = [
    segment_id
    for segment_id in concept_prompts.keys()
    if segment_id not in {seg["segment_id"] for seg in stage1_segments}
]

if missing_concepts:
    raise ValueError(f"Missing concept prompts: {missing_concepts}")

if extra_concepts:
    console.print(f"[yellow]Ignoring extra concept prompt keys: {extra_concepts}[/yellow]")
    expected = {seg["segment_id"] for seg in stage1_segments}
    concept_prompts = {
        key: value
        for key, value in concept_prompts.items()
        if key in expected
    }

# Preserve stage1 order.
concept_prompts = {
    seg["segment_id"]: concept_prompts[seg["segment_id"]]
    for seg in stage1_segments
}

prompt_pipeline.save_json(concept_prompts_json, concept_prompts)
log_file("Concept Prompts JSON", concept_prompts_json)
