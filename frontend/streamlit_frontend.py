import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Sentiment Analysis App")
st.write("Enter a movie review and the model will predict the sentiment.")

# Input text
review = st.text_area(
    "Enter Review",
    height=150,
    placeholder="Type your movie review here..."
)

# Button
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"review": review}
            )

            if response.status_code == 200:
                result = response.json()

                sentiment = result["sentiment"]

                if sentiment == "Positive":
                    st.success(f"Sentiment: {sentiment} 😀")
                else:
                    st.error(f"Sentiment: {sentiment} 😡")

                st.write("Prediction Label:", result["prediction_label"])

            else:
                st.error("API Error")

        except Exception as e:
            st.error("Could not connect to FastAPI server")