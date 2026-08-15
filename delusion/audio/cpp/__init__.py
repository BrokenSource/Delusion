import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import huggingface_hub
from pydantic import BaseModel, Field

import delusion

MODELS: Path = Path(os.getenv("HF_HOME", delusion.dirs.user_data_path.joinpath("audio-cpp")))
"""Global models directory"""

# ---------------------------------------------------------------------------- #

class AudioCPP(BaseModel, ABC):

    task: str = cast(str, None)
    """Supported audio processing modes"""

    family: str = cast(str, None)
    """Model family type"""

    # -------------------------------- |
    # Runtime options

    host: Annotated[str, Field(exclude=True, frozen=True)] = os.getenv("AUDIOCPP_HOST", "127.0.0.1:8080")
    """Server address (URL, IPv4, IPv6, localhost, hostname, ...)"""

    backend: Literal["cpu", "cuda", "rocm", "vulkan", "metal", "best"] = "best"
    """Hardware accelerator to use (support at compilation time)"""

    device: int = 0
    """Backend device index"""

    threads: int = 4
    """OpenMP worker threads"""

    # -------------------------------- |
    # Generation options

    seed: int | None = None
    """Generation seed when supported"""

    # -------------------------------- |
    # Model sources

    repository: str = "audio-cpp/audio.cpp-gguf"
    """HuggingFace source repository for models"""

    revision: str = "c3857f1ec35cfea8993924e7c2a6f682b5dc060b"
    """HuggingFace repository commit for caching"""

    @abstractmethod
    def filename(self) -> Path:
        """Model file location on repository"""
        ...

    def model(self) -> Path:
        """Path to the local downloaded model"""
        return MODELS.joinpath(self.filename())

    def download(self) -> Self:
        """Download model from HuggingFace"""
        huggingface_hub.hf_hub_download(
            repo_id=self.repository,
            filename=self.filename().as_posix(),
            revision=self.revision,
            local_dir=MODELS,
        )
        return self

# ---------------------------------------------------------------------------- #

class OmniVoice(AudioCPP):
    family: Literal["omnivoice"] = "omnivoice"

    task: Literal["tts"] = "tts"

    quant: Literal["q8_0", "bf16", "f16"] = "q8_0"
    """Model quantization"""

    def filename(self) -> Path:
        return Path("OmniVoice-GGUF", f"omnivoice-{self.quant}.gguf")

# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    omni = OmniVoice().download()
