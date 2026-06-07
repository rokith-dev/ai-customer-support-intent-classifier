import pickle

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

df = pd.read_csv(
    "data/processed/cleaned_tickets.csv"
)

X = df["issue_description"]

y = df["category"]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

print("\nSample Predictions:")

for actual, predicted in zip(y_test[:10], predictions[:10]):
    print(f"Actual: {actual} | Predicted: {predicted}")

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

with open("models/intent_classifier.pkl", "wb") as file:
    pickle.dump(model, file)

with open("models/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel and Vectorizer saved successfully!")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nUnique Categories:")
print(y.unique())

print("\nCategory Counts:")
print(y.value_counts())

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

print(y.value_counts())

print(y.unique())