import re
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    """
    clean the text by removing special characters and converting to lowercase
    """
    text = re.sub (r"<.*?>", "", text)  # remove HTML tags
    text = re.sub (r"[^a-zA-Z']", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text)

    return text 

def preprocess_dataset(dataset):
    texts = [clean_text(x['text']) for x in dataset]
    labels = [x['label'] for x in dataset]
    return  texts, labels 


def get_vectorizer():
    return TfidfVectorizer(max_features = 5000,  ngram_range = (1,2))