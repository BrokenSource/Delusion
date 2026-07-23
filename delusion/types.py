from typing import Literal

type Token = int
"""The gambling currency"""

type Second = float
"""The patience currency"""

type Role = Literal[
    "assistant",
    "system",
    "tool",
    "user",
]
