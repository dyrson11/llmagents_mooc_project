import re


def fix_apostrophes_and_quotes(text):
    output = list(text)
    quote_indices = []

    for i, char in enumerate(text):
        if char != "'":
            continue

        prev = text[i - 1] if i > 0 else ""
        next = text[i + 1] if i + 1 < len(text) else ""

        # Rule 1: Surrounded by letters → apostrophe
        if prev.isalpha() and next.isalpha():
            output[i] = "’"
        # Rule 2: Preceded by space/punct, followed by letter → apostrophe at word-start
        elif not next.isalpha():
            output[i] = '"'
        # Rule 3: Else treat as a quote, matching in pairs
        elif next.isupper():
            output[i] = '"'
        else:
            output[i] = "’"
            quote_indices.append(i)
    text = "".join(output)
    text = re.sub(r"\s*’\s*", "’", text)
    return text


def clean_spaces(text):
    """
    Elimina espacios innecesarios en el texto.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text
