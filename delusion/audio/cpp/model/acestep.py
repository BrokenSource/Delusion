from collections.abc import Iterable
from pathlib import Path
from typing import Literal, override

from delusion.audio.cpp.model import (
    AudioModel,
    SupportsMusicGeneration,
)
from delusion.audio.cpp.task import GenerationRequest, Task


class AceStep(
    AudioModel,
    SupportsMusicGeneration,
):
    family: Literal["ace_step"] = "ace_step"
    """Model architecture"""

    variant: Literal["base", "turbo"] = "base"
    """Model variant"""

    quant: Literal["q8_0", "bf16"] = "q8_0"
    """Model quantization"""

    @override
    def files(self) -> Iterable[Path]:
        yield Path("ACE-Step1.5-GGUF", self.variant, f"ace-step-1.5-{self.variant}-{self.quant}.gguf")

    @override
    def music(self,
        prompt: str,
        duration: float = 60,
        lyrics: str | None = None,
    ) -> GenerationRequest:
        return GenerationRequest(
            task=Task.Generation,
            family=self.family,
            model=self.model(),
            route="text2music",
            text=prompt,
            duration=duration,
            lyrics=lyrics,
        )
