> [!IMPORTANT]
> Work in progress - internal code refactor.

<div align="center">
  <h1>Delusion</h1>
  <span>✨ The missing conveniences in generative models ✨</span>
</div>

## 📦 Description

A small toolkit for generative models, with practical conveniences built in: such as caching, fast imports, syntactic sugars, opinionated architecture, curated providers, and more.

- [x] **Message** classes with type-safe structured output generics, auto validation (chat)
- [x] **Modular**: Easily write your implementations or modify existing ones.
- [x] **Minimal**: Avoids the complexity and commitment of a full framework.

## 📦 Usage

Simply add the [`delusion`](https://pypi.org/project/delusion/) PyPI package to your `pyproject.toml` and use it:

```toml
[project]
dependencies = ["delusion"]
```

### Chat

```python
from delusion.chat.router import Ollama
from pydantic import BaseModel, Field

class Country(BaseModel):
    name: str
    capital: str
    languages: set[str] = Field(
        description="Officially recognized languages"
    )

chat = Ollama().serve()
chat.gemma4("e2b").pull()

chat.send("Tell me about Canada, its capital and spoken languages.")

# Your linter should properly point to the class
canada = chat.generate(schema=Country)
print(canada.struct)

assert (canada.struct.name == "Canada")
assert (canada.struct.capital == "Ottawa")
assert (canada.struct.languages == {"English", "French"})
```
