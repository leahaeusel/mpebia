"""Test output utilities."""

from mpebia.output import get_directory


def test_get_directory(tmp_path):
    """Test getting the directory of a file."""
    file = tmp_path / "test_file.py"
    directory = get_directory(file)

    assert directory == tmp_path
