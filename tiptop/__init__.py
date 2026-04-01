from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tiptop")
except PackageNotFoundError:
    __version__ = "unknown"
