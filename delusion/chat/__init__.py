import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Literal, Self, cast

from pydantic import BaseModel, Field, computed_field

from delusion.types import Role, Second, Token

# ---------------------------------------------------------------------------- #

class Message[T: BaseModel](BaseModel):

    role: Role = "user"
    """Sender or message type"""

    think: str | None = None
    """Internal model reasoning"""

    content: str | None = None
    """Text content"""

    struct: T = cast(T, None)
    """Validated model instance"""

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

# ---------------------------------------------------------------------------- #

class Chat(BaseModel, ABC):

    model: str = cast("", None) # type: ignore
    """Common model name or identifier"""

    think: Literal["low", "medium", "high"] | bool = True
    """Whether to enable internal reasoning and/or its effort level"""

    messages: list[Message] = Field(default_factory=list)
    """Chat messages history"""

    @contextlib.contextmanager
    def branch(self) -> Generator[Self, None, None]:
        """Restores messages at entry on context exit"""
        self.messages = copy.deepcopy(this := self.messages)
        try:
            yield self
        finally:
            self.messages = this

    @contextlib.contextmanager
    def fork(self) -> Generator[Self]:
        """Get a new self-instance with current settings for multithreading"""
        yield type(self)(**self.model_dump())

    @abstractmethod
    def generate[T: BaseModel](self,
        schema: type[T] | None = None,
        retries: int = 3,
    ) -> Message[T]:
        """
        Generate the next message for chat context:
        - Ensures a valid schema when provided (N retries)
        - Appends the message to chat history
        """
        ...

    def send(self,
        content: str,
        role: Role = "user",
    ) -> Self:
        """Add a message to the chat history"""
        self.messages.append(Message(role=role, content=content))
        return self
