import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib
from src.preprocess_dataset.preprocess import preprocess
from src.features.vectorizer import tfidf

print("Started Sentiment Analysis Training Pipeline...")

print("Loading dataset...")
df = pd.read_csv("C:\\Users\\kousi\\Documents\\sentiment_analysis_fastapi_aws_docker\\dataset\\IMDB Dataset.csv")
print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

print("Starting preprocessing...")
df = preprocess(df)
print("Preprocessing completed")

print("Separating features and labels...")
X = df['review']
Y = df['sentiment']
print("Features and labels separated")

# Example sentiment labels
y = ["positive", "negative"]

print("Initializing Label Encoder...")
le = LabelEncoder()

print("Encoding sentiment labels...")
y_encoded = le.fit_transform(Y)
print("Label encoding completed")

print("Performing Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print("Train-Test Split completed")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Starting TF-IDF vectorization for training data...")
X_train_tfidf = tfidf.fit_transform(X_train)
print("TF-IDF vectorization completed for training data")

print("Transforming test data using trained TF-IDF...")
X_test_tfidf = tfidf.transform(X_test)
print("TF-IDF transformation completed for test data")

print("Initializing Logistic Regression model...")
model = LogisticRegression(max_iter=1000)

print("Starting model training...")
model.fit(X_train_tfidf, y_train)
print("Model training completed")

print("Generating predictions for training data...")
y_train_pred = model.predict(X_train_tfidf)

print("Generating predictions for testing data...")
y_test_pred = model.predict(X_test_tfidf)

print("Calculating accuracy scores...")
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred))

print("Preparing model directory...")
model_dir = "C:\\Users\\kousi\\Documents\\sentiment_analysis_fastapi_aws_docker\\models"

os.makedirs(model_dir, exist_ok=True)
print("Model directory ready")

print("Saving trained model...")
joblib.dump(model, os.path.join(model_dir, "sentiment_model.pkl"))

print("Saving TF-IDF vectorizer...")
joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

print("Model and vectorizer saved successfully")
print("Training pipeline completed successfully")