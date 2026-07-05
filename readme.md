A minimal generative models router, caching, structured output with native python linting.

Work in progress.

## Chat example

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
