import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(
    "data/processed/cleaned_tickets.csv"
)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    df["issue_description"]
)

print(X.shape)