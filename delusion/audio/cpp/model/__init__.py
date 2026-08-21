from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from pydantic import BaseModel

import delusion
from delusion.audio.cpp.task import GenerationRequest, GenerationResponse

MODELS: Path = Path(delusion.dirs.user_data_path.joinpath("audio-cpp"))
"""Global models directory"""


class AudioModel(BaseModel, ABC):

    family: str # Children must pin literals
    """Model architecture"""

    repository: str = "audio-cpp/audio.cpp-gguf"
    """HuggingFace source repository for models"""

    revision: str = "c3857f1ec35cfea8993924e7c2a6f682b5dc060b"
    """HuggingFace repository commit for caching"""

    @abstractmethod
    def filename(self) -> Path:
        """Model location on repository"""
        ...

    def download(self) -> Self:
        """Download model from HuggingFace"""
        import huggingface_hub
        huggingface_hub.hf_hub_download(
            repo_id=self.repository,
            filename=self.filename().as_posix(),
            revision=self.revision,
            local_dir=MODELS,
        )
        return self

    def model(self) -> Path:
        """Path to the local downloaded model"""
        return MODELS.joinpath(self.filename())

    def args(self) -> Iterable[str]:
        yield from ("--model", self.model())
        yield from ("--family", self.family)

# ---------------------------------------------------------------------------- #

class SupportsTextToSpeech(ABC):

    @abstractmethod
    def tts(self,
        text: str,
    ) -> GenerationRequest:
        """Generate audio for a text"""
        ...

class SupportsVoiceDesign(ABC):

    @abstractmethod
    def design(self,
        text: str,
        voice: str,
    ) -> GenerationRequest:
        """Voices from prompts"""
        ...
