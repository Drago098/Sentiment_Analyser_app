from sklearn.metrics import classification_report, confusion_matrix
import joblib
from src import config  # Assuming config contains paths to the model and vectorizer
from src.preprocess import preprocess_dataset  # Assuming these functions exist
from src.dataloader import load_imdb_dataset

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    # Load the saved model and vectorizer
    model = joblib.load(config.MODEL_PATH)
    vectorizer = joblib.load(config.VECTORIZER_PATH)

    # Load and preprocess the test dataset
    _, test_set = load_imdb_dataset()
    X_test, y_test = preprocess_dataset(test_set)
    X_test = vectorizer.transform(X_test)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

