import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch.nn.functional as F

#label_encoder = joblib.load("C:\\Users\\Rudra Thakar\\Jupyter\\PRML-Project-25\\label_encoder_category.pkl")
#scaler = joblib.load("scaler_fake.pkl")
#web_encoder = joblib.load("web_encoder_fake.pkl")  # adjust the path
#category_encoder_NB = joblib.load("C:\\Users\\Rudra Thakar\\Jupyter\\PRML-Project-25\\category_label_encoder.pkl")

class TunedCategoryANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TunedCategoryANN, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        #self.dropout3 = nn.Dropout(0.3)

        # self.fc4 = nn.Linear(128, 64)
        # self.bn4 = nn.BatchNorm1d(64)
        #self.dropout4 = nn.Dropout(0.3)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        # x = self.dropout3(x)

        #x = F.relu(self.bn4(self.fc4(x)))
        #x = self.dropout4(x)

        return self.out(x)  # Raw logits (used with CrossEntropyLoss)

model2 = TunedCategoryANN(416, 8)
model2.load_state_dict(torch.load("ann_categorgy_1.pth", map_location=torch.device('cpu')))
model2.eval()

dt_model = joblib.load("DT_Category.pkl")
naive_bayes_model = joblib.load("naive_bayes_category_model.pkl")
svm_model = joblib.load("svm_model.pkl")
randomForestModel = joblib.load("rf_model.pkl")
xgboost_model = joblib.load("xgb_category_model_final.pkl")

dt_stack_model = joblib.load("stack_dt_base.pkl")
svm_stack_model = joblib.load("stack_svm_base.pkl")
#randomForestModel = joblib.load("C:\\Users\\Rudra Thakar\\Jupyter\\PRML-Proxgboost_category_model_final.pkl")

class ann_stack_base(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ann_stack_base, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        return self.out(x)

ann_base_model = ann_stack_base(417, 9)  # input_size = 416, num_classes = 9
ann_base_model.load_state_dict(torch.load("stack_ann_base.pth", map_location=torch.device('cpu')))
ann_base_model.eval()


class StrongerMetaANN(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=32, output_dim=9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return self.out(x)  # raw logits

meta_ann_model = StrongerMetaANN()  # input_size = 416, num_classes = 9
meta_ann_model.load_state_dict(torch.load("meta__ann_model.pth", map_location=torch.device('cpu')))
meta_ann_model.eval()






def predict_category(input_tensor, model_choice):
    if model_choice == "ANN":
        with torch.no_grad():
            output_category = model2(input_tensor)  # shape: [1, 9]
            category_probs = torch.softmax(output_category, dim=1).numpy()  # Convert to NumPy

        return category_probs  # shape: (1, 9)

    elif model_choice == "Decision Tree":
        prob_dt = dt_model.predict_proba(input_tensor)  # shape: (1, 9)
        return prob_dt

    elif model_choice == "Naive Bayes":
        prob_nb = naive_bayes_model.predict_proba(input_tensor)  # shape: (1, 9)
        return prob_nb

    elif model_choice == "SVM":
        prob_svm = svm_model.predict_proba(input_tensor)  # shape: (1, 9)
        return prob_svm

    elif model_choice == "XGBoost":
        prob_xgb = xgboost_model.predict_proba(input_tensor)  # shape: (1, 9)
        return prob_xgb

    elif model_choice == "Random Forest":
        prob_rf = randomForestModel.predict_proba(input_tensor)  # shape: (1, 9)
        return prob_rf

    elif model_choice == "Stacked Model":
        dt_probs = dt_stack_model.predict_proba(input_tensor)  # shape: (1, 9)
        svm_probs = svm_stack_model.predict_proba(input_tensor)
        ann_base_model.eval()
        with torch.no_grad():
            ann_logits = ann_base_model(input_tensor)
            ann_probs = F.softmax(ann_logits, dim=1).cpu().numpy()

        combined_input = np.concatenate([dt_probs, svm_probs, ann_probs], axis=1)
        combined_tensor = torch.tensor(combined_input, dtype=torch.float32)

        meta_ann_model.eval()
        with torch.no_grad():
            meta_logits = meta_ann_model(combined_tensor)
            meta_probs = F.softmax(meta_logits, dim=1).cpu().numpy().flatten()
        return meta_probs.reshape(1, -1)

    else:
        return np.zeros((1, 9))  # Safe fallback for invalid model
