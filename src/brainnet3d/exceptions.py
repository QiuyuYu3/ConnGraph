"""
Exceptions raised by brainnet3d for problems with user data or downloads.
"""


class BrainNet3DError(Exception):
    """Base class for brainnet3d errors."""


class DataValidationError(BrainNet3DError, ValueError):
    """Input matrices or node tables are malformed or contain unusable values."""


class DownloadError(BrainNet3DError, RuntimeError):
    """A remote resource could not be downloaded or failed verification."""
