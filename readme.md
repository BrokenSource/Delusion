<div align="center">
  <img src="https://raw.githubusercontent.com/BrokenSource/Delusion/main/website/assets/logo.png" width="210">
  <h1>Delusion</h1>
  <span>✨ The missing conveniences in generative models ✨</span>
</div>

## 📦 Description

A small toolkit for generative models, with practical conveniences built in: such as caching, fast imports, syntactic sugars, opinionated architecture, curated providers, and more.

- [x] **Message** classes with type-safe structured output generics, auto validation (chat)
- [x] **Modular**: Easily write your implementations or modify existing ones.
- [x] **Minimal**: Avoids the complexity and commitment of a full framework.

## 📦 Usage

Simply add the [`delusion`](https://pypi.org/project/delusion/) PyPI package to your project and use it:

```toml
[project]
dependencies = ["delusion"]
```

### Chat

```python
from delusion.chat.router.ollamax import Ollama
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

# Your linter should properly point to the class
canada = chat.generate(schema=Country)
print(canada.model)

assert (canada.model.name == "Canada")
assert (canada.model.capital == "Ottawa")
assert (canada.model.languages == {"English", "French"})
```

## 📦 Standards

In an effort to minimize [xkcd 927](https://xkcd.com/927/), Delusion only introduces abstractions that provide clear value or represent shared semantics across providers, proxying or extending native packages whenever available.

_For example, the Options class for ollama models shall only apply to itself:_

```python
import ollama

local = Ollama(model="gemma4:e2b")
cloud = OpenAI(model="gpt-whatever")

isinstance(local.options, ollama.Options) # True
isinstance(cloud.options, ollama.Options) # False
```

Although both support `.temperature = 0.0`, the intended use is:

```python
# Individual settings
if os.getenv("PRODUCTION", None):
    chat = OpenAI(model=...)
    chat.options.temperature = 0.0
else:
    chat = Ollama(model=...)
    chat.options.temperature = 0.0

# Shared interface
chat.send(...)
chat.generate(schema=...)
```

Same for models: rather than over-abstracting capabilities, quantization, names, variants, and other provider-specific details, some code duplication is natural to keep it minimal and decoupled.

Conversely, `Message[T]` is abstracted because it represents a common semantic across providers.
