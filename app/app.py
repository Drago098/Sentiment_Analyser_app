import streamlit as st
from src.predict import predict_sentiment

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below, and we'll tell you if it's **Positive** or **Negative**.")

# Input box
user_input = st.text_area("Enter your review", height=150)

# Predict button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        label, confidence = predict_sentiment(user_input)
        sentiment = "👍 Positive" if label == 1 else "👎 Negative"
        st.success(f"Prediction: {sentiment} ({confidence * 100:.2f}% confidence)")
