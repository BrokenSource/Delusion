from pathlib import Path
from typing import Literal, override

from delusion.audio.cpp.model import (
    AudioModel,
    SupportsTextToSpeech,
)
from delusion.audio.cpp.task import GenerationRequest, Task


class OmniVoice(
    AudioModel,
    SupportsTextToSpeech,
):
    family: Literal["omnivoice"] = "omnivoice"
    """Model architecture"""

    quant: Literal["q8_0", "bf16", "f16"] = "q8_0"
    """Model quantization"""

    @override
    def filename(self) -> Path:
        return Path("OmniVoice-GGUF", f"omnivoice-{self.quant}.gguf")

    @override
    def tts(self, text: str) -> GenerationRequest:
        return GenerationRequest(
            task=Task.TextToSpeech,
            family=self.family,
            model=self.model(),
            text=text,
        )
