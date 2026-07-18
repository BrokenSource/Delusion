from dearlog import logger  # isort: split

from importlib.metadata import metadata

__meta__    = metadata(str(__package__))
__about__   = __meta__.get("Summary")
__version__ = __meta__.get("Version")

from smartdirs import Path, SmartDirs

dirs = SmartDirs(
    pkg=Path(__file__).parent,
    app=str(__package__),
    org="tremeschin",
    url="com",
)
