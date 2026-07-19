from dearlog import logger  # isort: split

__about__   = "✨ The missing conveniences in generative models"
__package__ = "delusion"
__version__ = "0.2.3"
__license__ = "MIT"

from smartdirs import Path, SmartDirs

dirs = SmartDirs(
    pkg=Path(__file__).parent,
    app=str(__package__),
    org="tremeschin",
    url="com",
)
