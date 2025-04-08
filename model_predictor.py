import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load resources
web_encoder = joblib.load("web_encoder_fake.pkl")
scaler = joblib.load("scaler_fake.pkl")
label_encoder = joblib.load("label_encoder_category.pkl")  # for SVM labels

bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ANN model definition
class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Load models
input_size = scaler.mean_.shape[0]
ann_model = ANNModel(input_size)
ann_model.load_state_dict(torch.load("ann_model_fake_news.pth", map_location=torch.device("cpu")))
ann_model.eval()

dt_model = joblib.load("DT_TrueFake.pkl")
svm_model = joblib.load("svm_model.pkl")

def extract_date_features(month_year_str):
    try:
        date = pd.to_datetime(month_year_str, format='%b-%y')
        return pd.Series({
            'year': date.year,
            'month': date.month,
            'quarter': date.quarter
        })
    except:
        return pd.Series({'year': 0, 'month': 0, 'quarter': 0})

def preprocess_input(statement, web, date_str, for_torch=True):
    # BERT embeddings
    bert_embedding = bert_model.encode([statement])[0]

    # One-hot encode source website
    web_encoded = web_encoder.transform([[web]])
    web_encoded = web_encoded.toarray()[0] if hasattr(web_encoded, 'toarray') else web_encoded[0]

    # Date features
    date_features = extract_date_features(date_str).fillna(0)

    # Combine all features
    combined_features = np.hstack([bert_embedding, web_encoded, date_features.values])
    combined_features = np.nan_to_num(combined_features)

    # Scale features
    scaled = scaler.transform([combined_features])

    if for_torch:
        return torch.tensor(scaled, dtype=torch.float32)
    else:
        return scaled

def predict_news(statement, web, date, model_choice):
    if model_choice == "ANN":
        input_tensor = preprocess_input(statement, web, date, for_torch=True)
        with torch.no_grad():
            output = ann_model(input_tensor)
            confidence = output.item()
            prediction = "Fake" if confidence < 0.5 else "True"
            if prediction == "Fake":
                confidence = 1 - confidence
            return prediction, confidence

    elif model_choice == "Decision Tree":
        input_np = preprocess_input(statement, web, date, for_torch=False)
        prob = dt_model.predict_proba(input_np)[0]
        pred_index = np.argmax(prob)
        label_map = {0: "Fake", 1: "True"}
        pred_label = label_map[pred_index]
        confidence = prob[pred_index]
        if pred_label == "Fake":
            confidence = 1 - confidence
        return pred_label, confidence

    elif model_choice == "SVM":
        input_np = preprocess_input(statement, web, date, for_torch=False)
        prob_svm = svm_model.predict_proba(input_np)
        pred_index = np.argmax(prob_svm)
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        confidence = prob_svm[0][pred_index]
        return pred_label, confidence

    else:
        return "Invalid model selected", None
