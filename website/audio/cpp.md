---
title: AudioCPP
icon: material/language-cpp
---

!!! warning "Work in progress"
    Just recently started audio.cpp integration, still thinking architecture and expanding abstracted models!

Example using [audio.cpp](https://github.com/0xShug0/audio.cpp):

```python
from delusion.audio.cpp import AudioCPP, OmniVoice

# No PyTorch or ONNX required
audio = AudioCPP(
    model=OmniVoice(quant="q8_0").download()
)

# Uses NamedTemporaryFile internally
speak = audio.tts(text="お水はもう一杯もらえますか")
Path("output.wav").write_bytes(speak.wav)
```
