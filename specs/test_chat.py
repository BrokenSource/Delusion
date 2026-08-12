from delusion.chat.option.ollamax import Ollama
from pydantic import BaseModel, Field


class Country(BaseModel):
    name: str
    capital: str
    languages: set[str] = Field(
        description="Officially recognized languages"
    )

def test_ollama():
    chat = Ollama().diskcache().serve()
    chat.gemma4("e2b").pull()
    chat.send("Tell me about Canada, its capital and spoken languages.")
    canada = chat.generate(schema=Country)

    # Linters should know all fields
    assert isinstance(canada.model, Country)
    assert (canada.model.name == "Canada")
    assert (canada.model.capital == "Ottawa")
    assert (canada.model.languages == {"English", "French"})

if __name__ == "__main__":
    test_ollama()
