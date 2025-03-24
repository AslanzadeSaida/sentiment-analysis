# src/api.py
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict_sentiment

# Logging konfiqurasiyası
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Sentiment Analysis API")
logging.info("FastAPI tətbiqi yaradıldı")

class TextItem(BaseModel):
    text: str

@app.post("/predict")
def get_prediction(item: TextItem):
    logging.info("Yeni POST sorğusu alındı")
    logging.info(f"Gələn mətn: {item.text}")

    sentiment = predict_sentiment(item.text)
    logging.info(f"Təyin olunan sentiment: {sentiment}")

    return {"sentiment": sentiment}
