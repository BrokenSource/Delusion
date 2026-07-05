import contextlib
import copy
from abc import ABC, abstractmethod
from typing import Generator, Literal, Optional, Self, Union

from diskcache import Cache as DiskCache
from pydantic import BaseModel, Field

import delusion

# ---------------------------------------------------------------------------- #
# https://github.com/pydantic/pydantic/issues/11603#issuecomment-4624919538

def getstate(self: BaseModel) -> dict:
    return {"__dict__": self.__dict__}

BaseModel.__getstate__ = getstate # type: ignore

CHAT_CACHE: DiskCache = DiskCache(
    directory=delusion.dirs.user_data_path.joinpath("chat"),
    size_limit=(512 * 1024**2), # MB
)

# ---------------------------------------------------------------------------- #

type Tokens = int
type Seconds = float
type Role = Literal[
    "assistant",
    "system",
    "tool",
    "user",
]

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

    stats: Stats = Field(default_factory=Stats)
    """Generation statistics"""

    think: Optional[str] = None
    """Internal model reasoning"""

    content: Optional[str] = None
    """Text content"""

    struct: T = None # type: ignore
    """Structured model output"""

# ---------------------------------------------------------------------------- #

class ChatModel(BaseModel, ABC):

    model: str = None # type: ignore
    """Common model name or identifier"""

    think: Union[Literal["low", "medium", "high"], bool] = True
    """Whether to enable internal reasoning and/or its effort level"""

    messages: list[Message] = Field(default_factory=list)
    """Chat history"""

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

    @contextlib.contextmanager
    def branch(self) -> Generator[Self, None, None]:
        stale = copy.deepcopy(self.messages)
        try:
            yield self
        finally:
            self.messages = stale
