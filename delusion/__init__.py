from dearlog import logger  # isort: split

__about__   = "🔮 Convenient generative models pythonic wrappers"
__package__ = "delusion"
__version__ = "0.4.0"
__license__ = "MIT"

from pathlib import Path

from platformdirs import PlatformDirs

dirs = PlatformDirs(
    appname=__package__,
    ensure_exists=True,
    opinion=True,
)
