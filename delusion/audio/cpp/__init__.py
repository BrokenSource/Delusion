import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self, override

from pydantic import BaseModel, Field

import delusion
from delusion.audio.cpp.model import (
    AudioModel,
    SupportsMusicGeneration,
    SupportsTextToSpeech,
    SupportsVoiceDesign,
)
from delusion.audio.cpp.model.omnivoice import OmniVoice
from delusion.audio.cpp.options import Options
from delusion.audio.cpp.runtime import Local, Runtime
from delusion.audio.cpp.task import GenerationResponse

type AudioModels = OmniVoice


class AudioCPP(BaseModel):

    model: AudioModel | AudioModels
    """Model to use"""

    options: Options = Field(default_factory=Options)
    """Generation options"""

    runtime: Runtime = Field(default_factory=Local)

    def tts(self, text: str) -> GenerationResponse:
        if not isinstance(self.model, SupportsTextToSpeech):
            raise TypeError(f"{type(self)} model does not support tts")

        return self.runtime.run(
            request=self.model.tts(text=text),
            options=self.options,
        )

    def music(self,
        prompt: str,
        duration: float = 60,
        lyrics: str | None = None,
    ) -> GenerationResponse:
        if not isinstance(self.model, SupportsMusicGeneration):
            raise TypeError(f"{type(self)} model does not support music generation")

        return self.runtime.run(
            request=self.model.music(
                prompt=prompt,
                duration=duration,
                lyrics=lyrics,
            ),
            options=self.options,
        )
