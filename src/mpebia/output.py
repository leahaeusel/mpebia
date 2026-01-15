"""Utility functions for outputting data to files."""

from pathlib import Path


def get_directory(file):
    """Get the directory of the provided file.

    Args:
        file (str): File path.

    Returns:
        Path: Directory of the file.
    """
    directory = Path(file).parent
    return directory


def get_object_attributes(obj):
    """Get all attributes and their values of an object.

    Args:
        obj (object): Object of interest.

    Returns:
        dict: Object attributes and their values.
    """
    attributes = {
        attr: getattr(obj, attr)
        for attr in dir(obj)
        if not callable(getattr(obj, attr)) and not attr.startswith("__")
    }
    return attributes
