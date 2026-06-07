import pandas as pd

from text_cleaner import clean_text


df = pd.read_csv("data/raw/customer_support_tickets_200k.csv")

df = df[["issue_description", "category"]]

df = df.dropna()

#df = df.drop_duplicates()

df["issue_description"] = df["issue_description"].apply(clean_text)

df.to_csv(
    "data/processed/cleaned_tickets.csv",
    index=False
)

print("Preprocessing completed successfully!")