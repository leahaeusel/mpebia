"""Test logging utilities."""

from mpebia.logging import get_logger


def test_get_logger(tmp_path):
    """Test getting a logger."""
    file = tmp_path / "test_file.py"
    logger = get_logger(file)
    test_message = "Test message"
    logger.info(test_message)

    with open(file.with_suffix(".log"), "r", encoding="utf-8") as f:
        file_contents = f.read()

    assert f"INFO: {test_message}" in file_contents
