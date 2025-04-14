# app.py
import streamlit as st
from model_predictor import predict_news
from demo_web_search import start_web_search
import numpy as np
import joblib
import torch
from collections import Counter
from category_predict_final import predict_category

st.title("📰 Fake News Detector")
st.write("Enter a news article and choose a model to classify it as Real or Fake.")

text = st.text_area("Enter News Text", height=100)

source = st.text_area("Enter Source", height=100)

date = st.text_area("Enter time of news", height=100)





model_choice = st.selectbox("Select Model to classify news as True or False", ["ANN", "Decision Tree", "Random Forest", "Random Forest with PCA", "SVM"])

if st.button("Predict"):
    if not text:
        st.warning("Please enter some text.")
    else:
        final_label = None
        final_confidence = None
        vector_stack = start_web_search(text, 0)
        combined_label = []
        combined_confidence = []
        for i in range(vector_stack.shape[0]):
            input_vector = vector_stack[i]
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
            label, threshold = predict_news(input_tensor, model_choice)
            combined_label.append(label.lower())
            combined_confidence.append(threshold)

        true_count = combined_label.count("true")
        false_count = combined_label.count("false")

        if true_count > false_count:
            final_label = "True"
        elif false_count > true_count:
            final_label = "False"
        else:
            # If it's a tie, you can break it using average confidence or default
            final_label = "True" if np.mean(combined_confidence) >= 0.5 else "False"
        # Final confidence is just the average of all confidences
        final_confidence = np.mean(combined_confidence)


        if label and threshold is not None:
            st.success(f"Prediction using {model_choice}: {final_label}")
            st.info(f"Model confidence: {final_confidence:.4f}")
        else:
            st.error("Model not supported.")


model_choice_category = st.selectbox("Select Model to find the Category of News", ["ANN", "Decision Tree", "Random Forest", "SVM" , "Stacked Model", "XGBoost"])

if st.button("Categorize"):
    if not text:
        st.warning("Please enter some text.")
    else:
        flag = 0
        if model_choice_category == "Stacked Model":
            flag = 1

        vector_stack = start_web_search(text, flag)  # shape: (7, 416)
        print(vector_stack)
        all_prob_vectors = []

        for i in range(vector_stack.shape[0]):
            input_vector = vector_stack[i]
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)

            prob_vector = predict_category(input_tensor, model_choice_category)  # shape: (1, 9)
            all_prob_vectors.append(prob_vector)

        # Step 1: Average the probability vectors
        avg_prob_vector = np.mean(np.vstack(all_prob_vectors), axis=0).reshape(1, -1)  # shape: (1, 9)
        
        if model_choice_category == "Stacked Model":
            class_counts = {
                'COVID-19': 17790,
                'ELECTION': 8500,
                'GOVERNMENT': 6923,
                'MISLEADIND': 20000,   # assuming typo in encoding, treating as one
                'MISLEADING': 5000,
                'POLITICS': 6670,
                'TERROR': 4384,
                'TRAD': 28000,
                'VIOLENCE': 7578
            }

            # === Step 2: Normalize Inverse Frequencies ===
            total = sum(class_counts.values())
            class_weights = {k: total / v for k, v in class_counts.items()}
            max_weight = max(class_weights.values())
            normalized_weights = {k: v / max_weight for k, v in class_weights.items()}

            # === Step 3: Map Weights to Label Indices ===
            # Encoding: 0 → COVID-19, 1 → ELECTION, ..., 8 → VIOLENCE
            ordered_class_names = ['COVID-19', 'ELECTION', 'GOVERNMENT', 'MISLEADIND', 'MISLEADING', 'POLITICS', 'TERROR', 'TRAD', 'VIOLENCE']
            weight_vector = np.array([normalized_weights[class_name] for class_name in ordered_class_names]).reshape(1, -1)

            # === Step 4: Apply Class Weight Recalibration to avg_prob_vector ===
            adjusted_prob_vector = avg_prob_vector * weight_vector
            adjusted_prob_vector /= adjusted_prob_vector.sum()  # Renormalize to sum to 1

            # === Step 5: Final Prediction ===
            final_index = np.argmax(adjusted_prob_vector)
            final_confidence_cat = adjusted_prob_vector[0][final_index]
        else:
            # Step 2: Final prediction
            final_index = np.argmax(avg_prob_vector)
            final_confidence_cat = avg_prob_vector[0][final_index]

        # Decode final category name
        label_encoder = joblib.load("label_encoder_category.pkl")
        final_category = label_encoder.inverse_transform([final_index])[0].lower()


        if final_category and final_confidence_cat is not None:
            st.success(f"Prediction using {model_choice_category}: {final_category}")
            st.info(f"Model confidence: {final_confidence_cat:.4f}")
            #st.info(f"Prob Vectors : {all_prob_vectors}")
            #st.info(f"avg prob Vector : {avg_prob_vector}")
        else:
            st.error("Model not supported.")
