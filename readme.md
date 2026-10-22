<div align="center">
  <img src="https://raw.githubusercontent.com/BrokenSource/Delusion/main/website/assets/logo.png" width="210">
  <h1>Delusion</h1>
  <span>🔮 Convenient generative models pythonic wrappers 🔮</span>
</div>

## Description

A small toolkit for generative models, with practical conveniences built in: such as caching, fast imports, syntactic sugars, opinionated architecture, curated providers, and more.

- [x] **Message** classes with type-safe structured output generics, auto validation (chat)
- [x] **Modular**: Easily write your implementations or modify existing ones.
- [x] **Minimal**: Avoids the complexity and commitment of a full framework.

## Examples

### Chat

Chat history, schema-typed messages, auto-validation:

```python
from delusion.chat.option.ollama import Ollama
from pydantic import BaseModel, Field

class Country(BaseModel):
    name: str
    capital: str
    languages: set[str] = Field(
        description="Officially recognized languages"
    )

chat = Ollama().cache().serve()
chat.gemma4("e2b").pull()
chat.send("Tell me about Canada, its capital and spoken languages.")

# Linters know all fields
canada = chat.generate(schema=Country)

assert (canada.model.name == "Canada")
assert (canada.model.capital == "Ottawa")
assert (canada.model.languages == {"English", "French"})
```

### Audio

Using [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp), text-to speech with [OmniVoice](https://github.com/k2-fsa/OmniVoice/):

```python
from delusion.audio.cpp import AudioCPP, OmniVoice

audio = AudioCPP(
    model=OmniVoice().download(),
)

speak = audio.tts(text="お水はもう一杯もらえますか")
Path("/tmp/output.wav").write_bytes(speak.wav)
```

Music generation with [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3):

```python
from delusion.audio.cpp import AudioCPP, StableAudio3

audio = AudioCPP(
    model=StableAudio3(quant="q8_0").download(),
)

music = audio.music(
    prompt="piano, flute, inspirational, ambient",
    duration=60,
)

Path("/tmp/output.wav").write_bytes(music.wav)
```

## Getting started

<div align="center">
  <i>
    Visit the documentation at
    <a href="https://delusion.tremeschin.com/">https://delusion.tremeschin.com/</a>
  </i>
</div>
