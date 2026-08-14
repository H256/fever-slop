from __future__ import annotations

from feverslop.domain.full_auto import SongSpec
from feverslop.ports.llm import LLMPort
from feverslop.prompting.general_modules import GeneralPromptModules


class LLMSongBriefGenerator:
    def __init__(self, llm: LLMPort, *, modules=None):
        self.llm = llm
        self._modules = modules if modules is not None else GeneralPromptModules(llm)

    def generate(self, request) -> SongSpec:
        payload = {
            "idea": request.idea,
            "style": request.style,
            "duration_seconds": request.duration_seconds,
            "language": request.language,
            "bpm_override": request.bpm,
            "keyscale_override": request.keyscale,
        }
        data = self._modules.song_brief(payload).model_dump()
        return SongSpec(
            title=str(data["title"]).strip(),
            tags=str(data["tags"]).strip(),
            lyrics=str(data["lyrics"]).strip(),
            bpm=int(data.get("bpm") or request.bpm or 120),
            duration_seconds=float(request.duration_seconds),
            language=str(data.get("language") or request.language).strip(),
            keyscale=str(data.get("keyscale") or request.keyscale or "C major").strip(),
            visual_story_idea=str(data["visual_story_idea"]).strip(),
            visual_style=str(data["visual_style"]).strip(),
        )
