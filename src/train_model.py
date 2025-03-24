# src/train_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
import joblib

from src.preprocessing import clean_text

def train_sentiment_model(
    train_path: str = "data/train.csv",
    test_path: str = "data/test.csv",
    model_path: str = "model.joblib",
    encoding: str = "latin-1"  # <--- Add this parameter
):
    """
    Train a sentiment analysis model using train.csv and evaluate on test.csv.
    Saves the trained model to model.joblib.
    """
    # 1. Load the data with the specified encoding
    df_train = pd.read_csv(train_path, encoding=encoding)
    df_test = pd.read_csv(test_path, encoding=encoding)
    
    # 2. Preprocess
    df_train['clean_text'] = df_train['text'].apply(clean_text)
    df_test['clean_text'] = df_test['text'].apply(clean_text)
    
    # 3. Vectorize text
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(df_train['clean_text'])
    X_test = vectorizer.transform(df_test['clean_text'])
    
    df_train["sentiment"] = df_train["sentiment"].astype(str)
    df_test["sentiment"] = df_test["sentiment"].astype(str)

    y_train = df_train["sentiment"]
    y_test = df_test["sentiment"]
    
    # 4. Train model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("=== Evaluation on Test Set ===")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save vectorizer + model as a single object
    joblib.dump((vectorizer, clf), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # If you want to run directly, you can still call it with defaults:
    train_sentiment_model()
    # Or specify a different encoding like "cp1252", "utf-8", etc.
    # train_sentiment_model(encoding="cp1252")
