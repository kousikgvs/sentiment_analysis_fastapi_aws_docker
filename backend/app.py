from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

import sys, os
# Step up from backend/ to project root, then into src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess_dataset.preprocess import preprocess

app = FastAPI(title="Sentiment Analysis API")

# Load model and vectorizer
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model = joblib.load(os.path.join(BASE_DIR, "models", "sentiment_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))

class ReviewRequest(BaseModel):
    review: str

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):

    df = pd.DataFrame({
        "review": [request.review]
    })

    # preprocessing
    df = preprocess(df)

    # vectorization
    X = tfidf.transform(df["review"])

    # prediction
    prediction = model.predict(X)[0]

    # convert numeric label to sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "review": request.review,
        "prediction_label": int(prediction),
        "sentiment": sentiment
    }