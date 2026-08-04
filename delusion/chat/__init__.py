import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Self, cast

from pydantic import BaseModel, Field

from delusion.chat.message import Message
from delusion.types import Role


class Chat[T: BaseModel](BaseModel, ABC):

    model: str = cast(str, None)
    """Common model name or identifier"""

    messages: list[Message[T]] = Field(default_factory=list)
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
        """Get a new instance with current settings for multithreading"""
        yield type(self)(**self.model_dump())

    @abstractmethod
    def generate[S: BaseModel](self,
        schema: type[S] | None = None,
        retries: int = 3,
    ) -> Message[S]:
        """
        Generate the next message for chat context:
        - Ensures a valid schema when provided (N retries)
        - Appends the message to chat history
        """
        ...

    def send(self,
        text: str,
        role: Role = "user",
    ) -> Self:
        """Add a message to the chat history"""
        message = Message(role=role, text=text)
        self.messages.append(message)
        return self
