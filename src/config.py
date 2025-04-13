import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizers", "vectorizer.pkl")