import contextlib
import copy
from abc import ABC, abstractmethod
from typing import Generator, Literal, Optional, Self, Union

from pydantic import BaseModel, Field, computed_field

type Tokens = int
"""The gambling currency"""

type Seconds = float
"""The patience currency"""

type Role = Literal[
    "assistant",
    "system",
    "tool",
    "user",
]

# ---------------------------------------------------------------------------- #

class Message[T: BaseModel](BaseModel):

    role: Role = "user"
    """Sender or message type"""

    class Stats(BaseModel):

        duration: Seconds = 0.0
        """Time taken to generate the message"""

        generated: Tokens = 0
        """Number of output tokens generated in the response"""

        context: Tokens = 0
        """Number of input tokens in the prompt"""

        @computed_field
        @property
        def tokens_per_second(self) -> float:
            return round(self.generated / (self.duration + 1e-9), 2)

    stats: Stats = Field(default_factory=Stats)
    """Generation statistics"""

    think: Optional[str] = None
    """Internal model reasoning"""

    content: Optional[str] = None
    """Text content"""

    struct: T = None # type: ignore
    """Structured model instance"""

# ---------------------------------------------------------------------------- #

class ChatModel(BaseModel, ABC):

    model: str = None # type: ignore
    """Common model name or identifier"""

    think: Union[Literal["low", "medium", "high"], bool] = True
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

    @abstractmethod
    def generate[T: BaseModel](self,
        schema: Optional[type[T]]=None,
        retries: int=3,
    ) -> Message[T]:
        """
        Generate the next message for chat context:
        - Ensures a valid schema when provided (N retries)
        - Appends the message to chat history
        """
        ...

    def send(self,
        content: str,
        role: Role="user",
    ) -> Self:
        """Add a message to the chat history"""
        self.messages.append(Message(role=role, content=content))
        return self
