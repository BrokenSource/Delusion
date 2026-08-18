import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self, override

from pydantic import BaseModel, Field

import delusion

MODELS: Path = Path(os.getenv("HF_HOME", delusion.dirs.user_data_path.joinpath("audio-cpp")))
"""Global models directory"""

def optarg(*items: Any) -> Sequence[Any]:
    return [] if None in items else items

# ---------------------------------------------------------------------------- #

class GenerationResponse(BaseModel):
    ...

class TextToSpeechResponse(BaseModel):
    wav: bytes

# ---------------------------------------------------------------------------- #

class Capability(StrEnum):

    Alignment = "align"
    """Forced alignment"""

    Diarization = "diar"
    """Speaker diarization"""

    Generation = "gen"
    """Music or sounds from prompts"""

    Midi = "midi"
    """Midi transcription"""

    Recognition = "spk"
    """Speaker recognition"""

    Separation = "sep"
    """Source separation"""

    SingConversion = "svc"
    """Singing voice conversion"""

    SpeechToText = "s2s"
    """Speech transcription"""

    TextToSpeech = "tts"
    """Generate audio for a text"""

    VoiceCloning = "clon"
    """Voice cloning (replicate source)"""

    VoiceConversion = "vc"
    """Voice conversion (replace a speaker)"""

    VoiceDesign = "vdes"
    """Voices from prompts"""


class AudioModel(BaseModel, ABC):

    family: str # Children must pin literals
    """Model architecture"""

    repository: str = "audio-cpp/audio.cpp-gguf"
    """HuggingFace source repository for models"""

    revision: str = "c3857f1ec35cfea8993924e7c2a6f682b5dc060b"
    """HuggingFace repository commit for caching"""

    @abstractmethod
    def capabilities(self) -> set[Capability]:
        """Supported audio processing modes"""

    @abstractmethod
    def filename(self) -> Path:
        """Model file location on repository"""

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


class OmniVoice(AudioModel):
    family: Literal["omnivoice"] = "omnivoice"

    quant: Literal["q8_0", "bf16", "f16"] = "q8_0"
    """Model quantization"""

    @override
    def capabilities(self) -> set[Capability]:
        return {
            Capability.TextToSpeech,
            Capability.VoiceDesign,
            Capability.VoiceCloning,
        }

    @override
    def filename(self) -> Path:
        return Path("OmniVoice-GGUF", f"omnivoice-{self.quant}.gguf")


type AudioModels = OmniVoice

# ---------------------------------------------------------------------------- #

class Compute(BaseModel):

    host: Annotated[str, Field(exclude=True, frozen=True)] = os.getenv("AUDIOCPP_HOST", "127.0.0.1:8080")
    """Server address (URL, IPv4, IPv6, localhost, hostname, ...)"""

    backend: Literal["cpu", "cuda", "rocm", "vulkan", "metal", "best"] = "best"
    """Hardware accelerator to use (support at compilation time)"""

    device: int = 0
    """Backend device index"""

    threads: int = 4
    """OpenMP worker threads"""

    def args(self) -> Iterable[str]:
        yield from ("--backend", self.backend)
        yield from ("--device", str(self.device))
        yield from ("--threads", str(self.threads))


class Options(BaseModel):
    """Generation options"""

    seed: int | None = None
    """Generation seed when supported"""

# ---------------------------------------------------------------------------- #

class AudioCPP(BaseModel):

    model: AudioModel | AudioModels
    """Model to use"""

    options: Options = Field(default_factory=Options)
    """Generation options"""

    compute: Compute = Field(default_factory=Compute)
    """Compute options"""

    # Fixme: Some models use tts path for design, clone, etc.

    def tts(self,
        text: str,
        instruct: str | None = None,
        # reference: ...
    ) -> TextToSpeechResponse:
        if not Capability.TextToSpeech in self.model.capabilities():
            raise RuntimeError("Model does not support tts")

        with tempfile.NamedTemporaryFile(suffix=".wav") as output:
            output = Path(output.name)

            subprocess.check_call((
                "audiocpp_cli",
                *self.compute.args(),
                *self.model.args(),
                *optarg("--instruct", instruct),
                "--task", Capability.TextToSpeech,
                "--text", text,
                "--out", output,
            ))

            return TextToSpeechResponse(
                wav=output.read_bytes(),
            )

    def design(self,
        text: str,
        instruct: str
    ):
        ...

# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    audio = AudioCPP(
        model=OmniVoice().download(),
    )

    speak = audio.tts(text="お水はもう一杯もらえますか")
    Path("/tmp/output.wav").write_bytes(speak.wav)
