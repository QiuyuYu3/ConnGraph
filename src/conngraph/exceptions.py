"""
Exceptions raised by conngraph for problems with user data or downloads.
"""


class ConnGraphError(Exception):
    """Base class for conngraph errors."""


class DataValidationError(ConnGraphError, ValueError):
    """Input matrices or node tables are malformed or contain unusable values."""


class DownloadError(ConnGraphError, RuntimeError):
    """A remote resource could not be downloaded or failed verification."""
