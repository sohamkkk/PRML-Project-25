# app.py
import streamlit as st
# from model_predictor import predict_news

st.title("📰 Fake News Detector")
st.write("Enter a news article and choose a model to classify it as Real or Fake.")

text = st.text_area("Enter News Text", height=200)

source = st.text_area("Enter Source", height=200)

date = st.text_area("Enter time of news", height=200)

model_choice = st.selectbox("Select Model", ["KNN", "SVM", "ANN", "Decision Tree"])

if st.button("Predict"):
    if not text:
        st.warning("Please enter some text.")
    else:
        # result = predict_news(text, source, date, model_choice)
        # label = "Fake News" if result == 1 else "Real News"
        # st.success(f"Prediction using {model_choice}: {label}")
        st.write("successful")