"""
Unit tests for text preprocessing module.
"""

from src.preprocess import clean_text


def test_clean_text_lowercase() -> None:
    """
    Test that text is converted to lowercase.
    """
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_remove_symbols() -> None:
    """
    Test that symbols and numbers are removed.
    """
    assert clean_text("Win $$$ 100%") == "win"


def test_clean_text_empty_input() -> None:
    """
    Test empty string input.
    """
    assert clean_text("") == ""


def test_clean_text_invalid_input() -> None:
    """
    Test non-string input.
    """
    assert clean_text(123) == ""
