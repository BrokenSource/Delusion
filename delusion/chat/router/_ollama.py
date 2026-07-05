import copy
import os
import shutil
import subprocess
import time
from typing import Any, Optional, Self

import cachetools
import ollama
from pydantic import BaseModel, Field, ValidationError

from delusion import logger
from delusion.chat import CHAT_CACHE, ChatModel, Message


class Ollama(ChatModel):

    host: str = "10.0.0.64"
    """Server address (URL, IPv4, IPv6, localhost, hostname)"""

    port: int = 11434
    """Server or proxy port"""

    options: ollama.Options = Field(default_factory=ollama.Options)
    """Generation options"""

    @staticmethod
    def cache(client: ollama.Client | Any, ) -> ollama.Client:
        """Apply caching to generative or data querying ollama calls"""
        client.web_search = cachetools.cached(CHAT_CACHE)(client.web_search) # type: ignore
        client.web_fetch  = cachetools.cached(CHAT_CACHE)(client.web_fetch)  # type: ignore
        client.generate   = cachetools.cached(CHAT_CACHE)(client.generate)   # type: ignore
        client.chat       = cachetools.cached(CHAT_CACHE)(client.chat)       # type: ignore
        return client

    def serve(self) -> Self:
        """Ensure ollama server is running"""
        os.environ.setdefault("OLLAMA_HOST", f"{self.host}:{self.port}")
        os.environ.setdefault("OLLAMA_FLASH_ATTENTION", str(1))
        os.environ.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")
        os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(1))
        os.environ.setdefault("OLLAMA_MAX_QUEUE", str(1000))
        os.environ.setdefault("OLLAMA_NO_CLOUD", str(1))

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

            self.messages.append(message)
            return message
        else:
            raise RuntimeError(f"Failed to generate valid {schema} in {retries} attempts")

# Safe: Patch same name re-exports
Ollama.cache(ollama._client)
Ollama.cache(ollama)
