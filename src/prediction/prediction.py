import pandas as pd
import joblib
from src.preprocess_dataset.preprocess import preprocess

model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

sample_text = """
<p>OMG this movie was sooo good!!! 😂😂</p>
I can't------------ believe it lol. []/:::
Check this........... review here: https://www.imdb.com/title/tt0111161/
BRB watching it again. IMO one of the best movies ever!)()
"""

df_1 = pd.DataFrame({
    "review": [sample_text]
})

df_1 = preprocess(df_1)

X_sample = tfidf.transform(df_1["review"])

prediction = model.predict(X_sample)

print("Predicted Sentiment:", prediction[0])