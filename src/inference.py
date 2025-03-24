# src/inference.py
import joblib

def load_model(model_path: str = "model.joblib"):
    """
    Loads the trained vectorizer and model from disk.
    Returns (vectorizer, model).
    """
    vectorizer, model = joblib.load(model_path)
    return vectorizer, model

def predict_sentiment(text: str, model_path: str = "model.joblib"):
    """
    Predict the sentiment of a given text string using the saved model.
    """
    vectorizer, model = load_model(model_path)
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

# Optional: quick test if you run this directly
if __name__ == "__main__":
    sample_text = "I love this product, it's fantastic!"
    print(predict_sentiment(sample_text))
