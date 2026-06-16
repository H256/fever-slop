import whisper
from pathlib import Path
from noise_reduction import reduce_start_keep_length

vocals_file = Path("projects/testdata/output/vocals_ComfyUI_00056_.mp3")
clean_file = Path("projects/testdata/output/vocals_clean_ComfyUI_00056_.mp3")

reduce_start_keep_length(
    input_file=vocals_file,
    output_file=clean_file,
)

model = whisper.load_model("large")

result = model.transcribe(
    str(clean_file),
    language="german",
    task="transcribe",
    verbose=False,
    condition_on_previous_text=False,
    no_speech_threshold=0.75,
    logprob_threshold=-0.5,
    compression_ratio_threshold=2.0,
    temperature=0,
)

print(result["text"])
