---
title: Ollama
icon: simple/ollama
---

```python
from delusion.chat.option.ollamax import Ollama
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
