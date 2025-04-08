# model_predictor.py
import pickle
import torch
import numpy as np
from input_bert import preprocess_input

# Load all models globally
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

# Load ANN separately
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(self, input_size=768):
        super(ANNModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification
        )

    def forward(self, x):
        return self.fc(x)

ann_model = ANNModel()
ann_model.load_state_dict(torch.load("ann_model_fake_news.pth", map_location=torch.device("cpu")))
ann_model.eval()

def predict_news(text, source, date, model_choice):
    vector = preprocess_input(text,source, date)

    if model_choice == "KNN":
        return knn_model.predict(vector)[0]
    elif model_choice == "SVM":
        return svm_model.predict(vector)[0]
    elif model_choice == "Decision Tree":
        return dt_model.predict(vector)[0]
    elif model_choice == "ANN":
        tensor_vec = torch.tensor(vector, dtype=torch.float32)
        with torch.no_grad():
            outputs = ann_model(tensor_vec)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()
    else:
        return "Invalid model selected"
