import copy
import os
import shutil
import subprocess
import time
from collections.abc import MutableMapping
from typing import Any, Optional, Self

import cachetools
import ollama
from ollama import Client, Options
from pydantic import BaseModel, Field, ValidationError

from delusion import logger
from delusion.cache import CHAT_CACHE
from delusion.chat import ChatModel, Message


class Ollama(ChatModel):

    host: str = Field(os.getenv("OLLAMA_HOST", "127.0.0.1:11434"), exclude=True)
    """Server address (URL, IPv4, IPv6, localhost, hostname)"""

    options: Options = Field(default_factory=Options)
    """Generation options"""

    @staticmethod
    def cache(client: Client | Any, cache: MutableMapping) -> Client:
        """Apply caching to generative or data querying ollama calls"""
        client.web_search = cachetools.cached(cache)(client.web_search) # type: ignore
        client.web_fetch  = cachetools.cached(cache)(client.web_fetch)  # type: ignore
        client.generate   = cachetools.cached(cache)(client.generate)   # type: ignore
        client.chat       = cachetools.cached(cache)(client.chat)       # type: ignore
        return client

    def serve(self) -> Self:
        """Ensure ollama server is running"""
        for _attempt in range(40):
            try:
                ollama.ps()
                break
            except ConnectionError:
                if _attempt == 1:
                    if not shutil.which("ollama"):
                        raise RuntimeError("Ollama wasn't found in the system")
                    logger.info("Starting ollama server")
                    subprocess.Popen(
                        args=("ollama", "serve"),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                time.sleep(0.1)
        else:
            raise RuntimeError("Couldn't start ollama server")

        return self

    def pull(self) -> Self:
        """Ensure model is available"""
        for item in ollama.list().models:
            if item.model == self.model:
                break
        else:
            logger.info(f"Pulling ollama model: {self.model}")
            ollama.pull(self.model)
        return self

    def gemma4(self, variant: str) -> Self:
        """https://ollama.com/library/gemma4"""
        self.model = f"gemma4:{variant}"
        self.options.temperature = 1.0
        self.options.top_p = 0.95
        self.options.top_k = 64
        return self

    def context(self, k: int) -> Self:
        self.options.num_ctx = k*1024
        return self

    def temperature(self, t: float) -> Self:
        self.options.temperature = t
        return self

    def generate[T: BaseModel](self,
        schema: Optional[type[T]]=None,
        retries: int=3,
    ) -> Message[T]:
        history = copy.deepcopy(self.messages)
        options = copy.deepcopy(self.options)

        # Model best practices
        if "gemma4" in self.model:
            for item in history:
                item.think = None

        # Guidelines for schema
        if schema is not None:
            history.append(Message(
                role="system",
                content='\n'.join((
                    f"Output a minimal valid json for the context above.",
                    f"Schema reminder: {schema.model_json_schema()}"
                )),
            ))

        for attempt in range(retries):

            # Roll the seed for cached attempts
            options.seed = (options.seed or 0) + 1

            # Attempt to generate
            response = ollama.chat(
                model=self.model,
                think=self.think,
                options=options,
                format=(schema.model_json_schema() if schema else None),
                messages=[
                    ollama.Message(
                        role=item.role,
                        content=item.content,
                        thinking=item.think,
                    )
                    for item in history
                ],
            )

            # Vendor the response into our model
            message = Message[T](role="assistant")
            message.content         = response.message.content
            message.think           = response.message.thinking
            message.stats.duration  = (response.total_duration    or 0)/10e9
            message.stats.generated = (response.eval_count        or 0)
            message.stats.context   = (response.prompt_eval_count or 0)

            # Ensure a valid schema when provided
            if (schema is not None):
                if (message.content is None):
                    raise ValueError("Message content is None")
                try:
                    message.struct = schema.model_validate_json(message.content)
                except ValidationError:
                    logger.minor((
                        f"Failed to validate {schema.__name__} model, "
                        f"attempt ({attempt+1}/{retries}) • Content: {message.content}"
                    ))
                    continue
            break

        else:
            raise RuntimeError(f"Failed to generate valid {schema} in {retries} attempts")

        self.messages.append(message)
        return message

if os.getenv("DELUSION_OLLAMA_CACHE", "1") == "1":

    # Safe: Patch same name re-exports
    Ollama.cache(ollama._client, CHAT_CACHE) # type: ignore
    Ollama.cache(ollama, CHAT_CACHE) # type: ignore

    # https://github.com/pydantic/pydantic/issues/11603#issuecomment-4624919538
    def getstate(self: BaseModel) -> dict:
        return {"__dict__": self.__dict__}

    ollama.Message.__getstate__ = getstate # type: ignore
    ollama.Options.__getstate__ = getstate # type: ignore
