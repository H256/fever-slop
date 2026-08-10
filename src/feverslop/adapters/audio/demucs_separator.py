import gc
from pathlib import Path

import torch
import torchaudio

from demucs import pretrained
from demucs.apply import apply_model


class DemucsSeparator:
    def __init__(
        self,
        model_name: str = "htdemucs_6s",
        device: str = "auto",
        shifts: int = 2,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.shifts = shifts

        self.model = pretrained.get_model(model_name)
        self.model.to(device)
        self.model.eval()

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return

        self.model = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def separate(
        self,
        input_file: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        input_file = Path(input_file).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        waveform, sample_rate = torchaudio.load(str(input_file))

        # mono -> stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # mehr als 2 Kanäle -> erste zwei
        if waveform.shape[0] > 2:
            waveform = waveform[:2]

        target_sr = int(self.model.samplerate)

        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                target_sr,
            )
            sample_rate = target_sr

        mix = waveform.unsqueeze(0).to(self.device)

        with torch.no_grad():
            stems = apply_model(
                self.model,
                mix,
                device=self.device,
                progress=False,
                shifts=self.shifts,
            )

        stems = stems[0].cpu()

        outputs = {}

        for idx, stem_name in enumerate(self.model.sources):
            stem_audio = stems[idx]

            output_file = output_dir / f"{stem_name}_{input_file.stem}.wav"

            torchaudio.save(
                str(output_file),
                stem_audio,
                sample_rate,
                format="wav",
            )

            outputs[stem_name] = output_file

        return outputs
