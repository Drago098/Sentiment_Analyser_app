import joblib
from src import config
from src.preprocess import clean_text

def load_model_and_vectorizer():
    model = joblib.load(config.MODEL_PATH)
    vectorizer = joblib.load(config.VECTORIZER_PATH)
    return model, vectorizer

def predict_sentiment(review_text):
    model, vectorizer = load_model_and_vectorizer()
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0].max()  # Get confidence
    label = "Positive" if prediction == 1 else "Negative"
    return label, proba
