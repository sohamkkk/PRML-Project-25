import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load encoders and scalers
web_encoder = joblib.load("web_encoder_fake.pkl")
scaler = joblib.load("scaler_fake.pkl")

# Load BERT model
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define ANN model
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

# Load ANN model
input_size = scaler.mean_.shape[0]
ann_model = ANNModel(input_size)
ann_model.load_state_dict(torch.load("ann_model_fake_news.pth", map_location=torch.device("cpu")))
ann_model.eval()

# Load DT model
dt_model = joblib.load("DT_TrueFake.pkl")

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

def preprocess_input(statement, web, date_str):
    # BERT embedding
    bert_embedding = bert_model.encode([statement])[0]  # shape: (384,)
    
    # Encode website
    web_encoded = web_encoder.transform([[web]])
    web_encoded = web_encoded.toarray()[0] if hasattr(web_encoded, 'toarray') else web_encoded[0]
    
    # Date features
    date_features = extract_date_features(date_str).fillna(0)

    # Combine features
    combined_features = np.hstack([bert_embedding, web_encoded, date_features.values])
    combined_features = np.nan_to_num(combined_features)

    # Scale
    final_scaled = scaler.transform([combined_features])
    return torch.tensor(final_scaled, dtype=torch.float32)

def predict_news(statement, web, date, model_choice):
    input_tensor = preprocess_input(statement, web, date)

    if model_choice == "ANN":
        with torch.no_grad():
            output = ann_model(input_tensor)
            confidence = output.item()
            prediction = "Fake" if confidence < 0.5 else "True"
            if prediction == "Fake":
                confidence = 1 - confidence  # Flip for display
            return prediction, confidence

    elif model_choice == "Decision Tree":
        input_np = input_tensor.numpy()
        prob = dt_model.predict_proba(input_np)[0]  # shape: (2,)
        pred_index = np.argmax(prob)
        label_map = {0: "Fake", 1: "True"}
        pred_label = label_map[pred_index]
        confidence = prob[pred_index]

        if pred_label == "Fake":
            confidence = 1 - confidence  # Flip confidence for display

        return pred_label, confidence

    else:
        return "Invalid model selected", None

