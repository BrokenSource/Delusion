from typing import cast

from pydantic import BaseModel, Field, computed_field

from delusion.types import Role, Second, Token


class Message[T: BaseModel](BaseModel):

    role: Role = "user"
    """Sender or message type"""

    think: str | None = None
    """Internal model reasoning"""

    text: str | None = None
    """Text content"""

    model: T = cast(T, cast(object, None))
    """Validated schema model instance"""

    audio: list[None] = Field(default_factory=list)
    """Audio content"""

    image: list[None] = Field(default_factory=list)
    """Image content"""

    class Stats(BaseModel):

        duration: Second = 0.00
        """Time taken to generate the message"""

        generated: Token = 0
        """Number of output tokens generated in the response"""

        context: Token = 0
        """Number of input tokens in the prompt"""

        @computed_field
        @property
        def tokens_per_second(self) -> float:
            return (
                round(self.generated / self.duration, 2)
                if self.duration > 0 else 0.00
            )

    stats: Stats = Field(default_factory=Stats)
    """Generation statistics"""
