from src.preprocessing.text_cleaner import clean_text


def test_clean_text_removes_punctuation_and_normalizes_case():
    assert clean_text("Hello, WORLD!!") == "hello world"
