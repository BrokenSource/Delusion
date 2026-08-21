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
    task: Task

    family: str

    model: Path

    text: str | None = None

    instruct: str | None = None

    voice_ref: Path | None = None

    reference_text: str | None = None

    def args(self) -> Iterable[str]:
        yield from ("--task", self.task.value)
        yield from ("--model", self.model)
        yield from ("--family", self.family)

        if self.text is not None:
            yield from ("--text", self.text)


class GenerationResponse(BaseModel):
    wav: bytes
