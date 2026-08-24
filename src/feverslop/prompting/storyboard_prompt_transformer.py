from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from feverslop.ports.llm import LLMPort
from feverslop.prompting.general_modules import GeneralPromptModules


logger = logging.getLogger(__name__)


@dataclass
class TemplateStoryboardPromptTransformer:
    llm: LLMPort
    template_path: Path
    debug_dir: Path
    modules: object | None = None
    max_words: int = 150

    def __post_init__(self) -> None:
        if self.max_words < 1:
            raise ValueError("max_words must be >= 1")
        self._modules = self.modules if self.modules is not None else GeneralPromptModules(self.llm)

    def transform_prompt(
        self,
        *,
        scene_number: int,
        original_prompt: str,
        width: int,
        height: int,
    ) -> str:
        system_prompt, user_template = split_template(self.template_path.read_text(encoding="utf-8-sig"))
        user_prompt = (
            user_template
            .replace("{{width}}", str(width))
            .replace("{{height}}", str(height))
            .replace("{{original_prompt}}", original_prompt)
            .strip()
        )
        result = self._modules.storyboard_transform({
            "system_template": system_prompt,
            "user_template": user_prompt,
            "width": width,
            "height": height,
            "original_prompt": original_prompt,
            "max_words": self.max_words,
        })
        response = result.prompt.strip()
        words = response.split()
        if len(words) > self.max_words:
            logger.warning(
                "Storyboard prompt exceeded the %d-word limit (%d words); trimmed.",
                self.max_words,
                len(words),
            )
            response = " ".join(words[: self.max_words])
        self._write_debug(scene_number, system_prompt, user_prompt, response)
        return response

    def _write_debug(self, scene_number: int, system_prompt: str, user_prompt: str, response: str) -> None:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.debug_dir / f"scene_{scene_number:04}"
        (prefix.with_name(f"{prefix.name}_system.txt")).write_text(system_prompt, encoding="utf-8")
        (prefix.with_name(f"{prefix.name}_user.txt")).write_text(user_prompt, encoding="utf-8")
        (prefix.with_name(f"{prefix.name}_response.txt")).write_text(response, encoding="utf-8")


def split_template(template: str) -> tuple[str, str]:
    system_marker = "[SYSTEM]"
    user_marker = "[USER]"
    if system_marker not in template:
        raise ValueError("Storyboard prompt template is missing [SYSTEM]")
    if user_marker not in template:
        raise ValueError("Storyboard prompt template is missing [USER]")

    _before_system, after_system = template.split(system_marker, 1)
    system_prompt, user_template = after_system.split(user_marker, 1)
    return system_prompt.strip(), user_template.strip()
