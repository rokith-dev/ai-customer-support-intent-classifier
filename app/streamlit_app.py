from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Intent Classifier", page_icon="🎫", layout="wide")

st.title("AI Customer Support Intent Classifier")
st.write("This app will let you classify support tickets by intent once a trained model is available.")

sample_text = st.text_area("Enter a support ticket", placeholder="Type a customer support message here...")

if st.button("Predict intent"):
    if sample_text.strip():
        st.info("Model inference can be wired here after training artifacts are available.")
    else:
        st.warning("Please enter a ticket first.")
