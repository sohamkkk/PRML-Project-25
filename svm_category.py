
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

label_encoder = joblib.load("label_encoder_category.pkl")
svm_model = joblib.load("svm_model.pkl")

def extract_date_features(month_year_str):
    try:
        # Parse date in format like "Oct-20"
        date = pd.to_datetime(month_year_str, format='%b-%y')
        return pd.Series({
            'year': date.year,
            'month': date.month,
            'quarter': date.quarter
        })
    except:
        # Fallback in case of parsing error
        print("Except")
        return pd.Series({'year': 0, 'month': 0, 'quarter': 0})


def preprocess_input(statement, web, date_str):
    # STEP 1: BERT Embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    bert_embeddings = model.encode([statement])
    #print(bert_embeddings)

    # STEP 2.5: Apply Date Feature Extraction
    date_features = extract_date_features(date_str)

    # STEP 3: Source (Web) Encoding
    web_encoder = joblib.load("web_encoder_fake.pkl")  # adjust the path
    web_encoded = web_encoder.transform([[web]])  # double brackets to create 2D shape

    # Step 1: Replace NaNs in date features
    date_features_filled = date_features.fillna(0)
    date_features_array = date_features_filled.values.reshape(1, -1)

    # Step 3: Concatenate again
    final_features = np.hstack([
        bert_embeddings,
        web_encoded,
        date_features_array
    ])

    # Step 4: Replace any remaining NaNs (if any) as safety net
    final_features = np.nan_to_num(final_features, nan=0.0)
    #print(final_features)

    # Step 5: Normalize
    scaler = joblib.load("scaler_fake.pkl")

    # Use transform instead of fit_transform
    final_features_scaled = scaler.transform(final_features)

    return final_features_scaled

def svm_predict(statement, web, date):
    # Get preprocessed feature vector
    input_vec = preprocess_input(statement, web, date)

    # Predict probabilities
    prob_svm = svm_model.predict_proba(input_vec)  # shape: (1, num_classes)

    # Get index of highest probability
    pred_index = np.argmax(prob_svm)

    # Get corresponding label and confidence
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    confidence = prob_svm[0][pred_index]

    return pred_label, confidence

stmt = "PM modi hold high level meeting with the cabinet"
web = "NDTV"
date = "Jan-21"
label, conf = svm_predict(stmt, web, date)
print(f"Prediction: {label} (Confidence: {conf:.4f})")

