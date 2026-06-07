import pickle
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AI Customer Support Intent Classifier",
    page_icon="🤖",
    layout="wide"
)

# Load Model
with open("models/intent_classifier.pkl", "rb") as file:
    model = pickle.load(file)

# Load Vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Header
st.title("🤖 AI Customer Support Intent Classifier")

st.markdown("""
Automatically classify customer support tickets into predefined categories using
Natural Language Processing (NLP) and Machine Learning.

**Model:** Logistic Regression  
**Feature Engineering:** TF-IDF Vectorization
""")

st.divider()

# Input Section
st.subheader("📝 Customer Message")

user_input = st.text_area(
    "Enter customer issue",
    height=150,
    placeholder="Example: I forgot my password and cannot access my account..."
)

# Prediction Button
if st.button("🚀 Predict Intent"):

    if user_input.strip() == "":
        st.warning("Please enter a customer message.")
    else:

        text_vector = vectorizer.transform(
            [user_input.lower()]
        )

        prediction = model.predict(
            text_vector
        )

        st.success(
            f"Predicted Category: {prediction[0]}"
        )

st.divider()

# Sidebar
st.sidebar.header("📊 Supported Categories")

st.sidebar.write("""
- Login Issue
- Payment Problem
- Refund Request
- Security Concern
- Feature Request
- Performance Issue
- Bug Report
- Data Sync Issue
- Account Suspension
- Subscription Cancellation
""")

st.sidebar.header("🛠 Tech Stack")

st.sidebar.write("""
- Python
- Pandas
- Scikit-Learn
- TF-IDF
- Logistic Regression
- Streamlit
""")

# Footer
st.markdown("---")

st.caption(
    "Built as an AI Engineering & NLP Portfolio Project"
)