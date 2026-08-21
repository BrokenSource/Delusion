import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal, override

from pydantic import BaseModel, Field

from delusion.audio.cpp.options import Options
from delusion.audio.cpp.task import GenerationRequest, GenerationResponse


class Compute(BaseModel):
    """Runtime options"""

    backend: Literal["cpu", "cuda", "rocm", "vulkan", "metal", "best"] = "best"
    """Hardware accelerator to use (support at compilation time)"""

    device: int = 0
    """Backend device index"""

    threads: int = 4
    """OpenMP worker threads"""

    def args(self) -> Iterable[str]:
        yield from ("--backend", self.backend)
        yield from ("--device", str(self.device))
        yield from ("--threads", str(self.threads))


class Runtime(Compute, ABC):
    kind: str # Children must pin literals

    @abstractmethod
    def run(self,
        request: GenerationRequest,
        options: Options,
    ) -> GenerationResponse:
        ...


class Local(Runtime):
    kind: Literal["local"] = "local"

    cli: str = "audiocpp_cli"

    @override
    def run(self,
        request: GenerationRequest,
        options: Options,
    ) -> GenerationResponse:
        with tempfile.NamedTemporaryFile(suffix=".wav") as output:
            output = Path(output.name)

            subprocess.check_call((
                self.cli,
                *Compute.args(self),
                *options.args(),
                *request.args(),
                "--out", output,
            ))

            return GenerationResponse(
                wav=output.read_bytes()
            )


class Server(Runtime):
    kind: Literal["server"] = "server"

    host: Annotated[str, Field(exclude=True, frozen=True)] = os.getenv("AUDIOCPP_HOST", "127.0.0.1:8080")
    """Server address (URL, IPv4, IPv6, localhost, hostname, ...)"""


class Docker(Runtime):
    kind: Literal["docker"] = "docker"

    image: str = "0xshug0/audio.cpp:full-cuda13"
