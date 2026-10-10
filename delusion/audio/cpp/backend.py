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
    """Computation backend options"""

    backend: Literal["cpu", "cuda", "rocm", "vulkan", "metal", "best"] = "best"
    """Hardware accelerator to use (support depends on compile options)"""

    device: int = 0
    """Backend device index"""

    threads: int = (os.cpu_count() or 4)
    """OpenMP worker threads"""

    def args(self) -> Iterable[str]:
        yield from ("--backend", self.backend)
        yield from ("--device", str(self.device))
        yield from ("--threads", str(self.threads))


class Backend(BaseModel, ABC):
    kind: str # Children must pin literals

    @abstractmethod
    def run(self,
        request: GenerationRequest,
        options: Options,
    ) -> GenerationResponse:
        ...


class Local(Backend):
    kind: Literal["local"] = "local"

    compute: Compute = Field(default_factory=Compute)
    """Computation backend options"""

    cli: str = "audiocpp_cli"
    """Path to an AudioCPP CLI binary"""

    @override
    def run(self,
        request: GenerationRequest,
        options: Options,
    ) -> GenerationResponse:
        with tempfile.NamedTemporaryFile(suffix=".wav") as output:
            output = Path(output.name)

            subprocess.check_call((
                self.cli,
                *self.compute.args(),
                *options.args(),
                *request.args(),
                "--out", output,
            ))

            return GenerationResponse(
                wav=output.read_bytes()
            )


class Server(Backend):
    kind: Literal["server"] = "server"

    host: Annotated[str, Field(exclude=True, frozen=True)] = os.getenv("AUDIOCPP_HOST", "127.0.0.1:8080")
    """Server address (URL, IPv4, IPv6, localhost, hostname, ...)"""


class Docker(Backend):
    kind: Literal["docker"] = "docker"

    image: str = "0xshug0/audio.cpp:full-cuda13"
