from pathlib import Path

from delusion.audio.cpp import AudioCPP, OmniVoice

if __name__ == "__main__":
    audio = AudioCPP(
        model=OmniVoice().download(),
    )

    from rich import print
    print(AudioCPP.model_json_schema())
    print(audio.model_dump())

    speak = audio.tts(text="お水はもう一杯もらえますか")
    Path("/tmp/output.wav").write_bytes(speak.wav)
