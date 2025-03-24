# src/preprocessing.py
import re
import string
import pandas as pd

def clean_text(text):
    # If the value is NaN, or if it's float/integer, convert to empty string or coerce to string
    if pd.isna(text):
        return ""
    text = str(text)  # Convert anything to a string

    # Now apply your cleaning steps
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.strip()
    return text
