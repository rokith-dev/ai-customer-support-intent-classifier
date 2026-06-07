import pickle
import streamlit as st


with open("models/intent_classifier.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


st.title("AI Customer Support Intent Classifier")

st.write(
    "Enter a customer support message and predict its category."
)


user_input = st.text_area(
    "Customer Message"
)


if st.button("Predict"):

    text_vector = vectorizer.transform(
        [user_input.lower()]
    )

    prediction = model.predict(
        text_vector
    )

    st.success(
        f"Predicted Category: {prediction[0]}"
    )