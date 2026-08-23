"""
Stable Audio 3 model by Stability AI

Quite fast and decent, only for instrumental and sound effects generation.

# License

For commercial use, you must choose and agree to a community or enterprise
license from the model owners, as seen in (please check independently):

- https://stability.ai/license
- https://huggingface.co/collections/stabilityai/stable-audio-3
"""

from pathlib import Path
from typing import Literal, override

from delusion.audio.cpp.model import (
    AudioModel,
    SupportsMusicGeneration,
)
from delusion.audio.cpp.task import GenerationRequest, Task


class StableAudio3(
    AudioModel,
    SupportsMusicGeneration,
):
    family: Literal["stable_audio"] = "stable_audio"
    """Model architecture"""

    variant: Literal["small-music", "small-sfx", "medium"] = "medium"
    """Model variant"""

    quant: Literal["q8_0", "f16"] = "q8_0"
    """Model quantization"""

    @override
    def filename(self) -> Path:
        variant = {
            "small-music": "Small-Music",
            "small-sfx": "Small-SFX",
            "medium": "Medium",
        }[self.variant]

        return Path(
            f"Stable-Audio-3-{variant}-GGUF",
            f"stable-audio-3-{self.variant}-{self.quant}.gguf"
        )

    # Fixme: models may not accept all arguments,
    #   should we use inheritance? Stable(AudioCPP)
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
            text=prompt,
            duration=duration,
        )
