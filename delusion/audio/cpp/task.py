from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class Task(StrEnum):

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


class GenerationRequest(BaseModel):
    """Abstracts over all [non-runtime, non-generation] AudioCPP options"""

    task: Task
    """Internal task branch"""

    family: str
    """Model architecture family"""

    model: Path
    """Path to a model GGUF file"""

    route: str | None = None
    """Internal task route"""

    text: str | None = None
    """Varies by model - what to speak, a prompt, instruction, lyrics, etc."""

    instruct: str | None = None
    """Varies by model - usually voice design or generation guidance"""

    reference_audio: Path | None = None
    """Path to a reference audio, usually for initialization, cloning, extending"""

    reference_text: str | None = None
    """Reference audio transcription for alignment, better results"""

    lyrics: str | None = None
    """Music generation models singing words, when supported"""

    duration: float | None = None
    """Varies by model, target or precise generation duration"""

    def args(self) -> Iterable[str]:
        yield from ("--task", self.task)
        yield from ("--model", str(self.model))
        yield from ("--family", self.family)

        if self.route is not None:
            yield from ("--task-route", self.route)

        if self.text is not None:
            yield from ("--text", self.text)

        if self.duration is not None:
            yield from ("--duration-seconds", str(self.duration))

        if self.lyrics is not None:
            yield from ("--lyrics", self.lyrics)


class GenerationResponse(BaseModel):
    wav: bytes
