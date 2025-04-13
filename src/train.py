import joblib 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src import config 
from src.preprocess import preprocess_dataset, get_vectorizer
from src.dataloader import load_imdb_dataset
from sklearn.model_selection import train_test_split
import os

def train():
    train_set, test_set = load_imdb_dataset()
    X_train, y_train = preprocess_dataset(train_set)
    X_test, y_test = preprocess_dataset(test_set)
    vectorizer = get_vectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")



    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.VECTORIZER_PATH), exist_ok=True)

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    print("Model and vectorizer saved.")
if __name__ == "__main__":
    train()  # Ensures the function runs when executing the file
