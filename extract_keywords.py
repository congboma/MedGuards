import pandas as pd
from difflib import SequenceMatcher

# Read file
df = pd.read_csv("Dataset.csv")

def extract_keywords(error_sentence, corrected_sentence):
    """
    Find the new/different words added in the Corrected Sentence
    using difflib.SequenceMatcher for token-level diff.
    """

    # Tokenize
    error_tokens = str(error_sentence).split()
    corrected_tokens = str(corrected_sentence).split()

    matcher = SequenceMatcher(None, error_tokens, corrected_tokens)

    added_words = []

    # Traverse diff results
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            added_words.extend(corrected_tokens[j1:j2])

    # Remove duplicates while preserving order
    return " ".join(dict.fromkeys(added_words))

# Generate new column auto-keywords
df["auto-keywords"] = df.apply(
    lambda row: extract_keywords(row["Error Sentence"], row["Corrected Sentence"]),
    axis=1
)

# Save new file
df.to_csv("dataset_with_autokeywords.csv", index=False)

df.head()