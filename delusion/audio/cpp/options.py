from collections.abc import Iterable

from pydantic import BaseModel


class Options(BaseModel):
    """Generation options"""

    seed: int | None = None
    """Generation seed when supported"""

    def args(self) -> Iterable[str]:
        if self.seed is not None:
            yield from ("--seed", self.seed)
