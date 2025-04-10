import os
import numpy as np
import pandas as pd
import joblib
import warnings
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from simple_naive_bayes import SimpleNaiveBayes
warnings.filterwarnings("ignore")  

# this function loads all the models and encoders that we'll need
def load_models():
    return {
        "category_model": joblib.load("naive_bayes_category_model.pkl"),
        "category_vectorizer": joblib.load("category_vectorizer.pkl"),
        "category_encoder": joblib.load("category_label_encoder.pkl"),
        "label_model": joblib.load("naive_bayes_label_model.pkl"),
        "web_encoder": joblib.load("web_encoder_fake.pkl"),
        "scaler": joblib.load("scaler_fake.pkl"),
        "bert_model": joblib.load("bert_model1.pkl")
    }

# converts something like "Jan-21" into year, month, and quarter
def extractDateFeatures(monthYearStr):
    try:
        date = pd.to_datetime(monthYearStr, format='%b-%y')
        return pd.Series({
            'year': date.year,
            'month': date.month,
            'quarter': date.quarter
        })
    except Exception:
        # if parsing fails, just return 0s
        return pd.Series({'year': 0, 'month': 0, 'quarter': 0})

# here we prepare the full input vector for predicting fake/true
def preprocessLabelInput(statement, web, dateStr, web_encoder, scaler, bertModel):
    try:
        # get the BERT embedding for the input text
        bertEmbed = bertModel.encode([statement])  # gives a 384-dim vector
    except Exception as e:
        raise RuntimeError(f"Error in BERT embedding: {str(e)}")

    # get features like year, month, quarter from date string
    dateFeat = extractDateFeatures(dateStr).fillna(0).values.reshape(1, -1)

    # encode the source website (like NDTV, etc.)
    webEncoded = web_encoder.transform(pd.DataFrame([[web]], columns=['Web']))
    webEncoded = webEncoded * 0.1  # scaling it down a bit

    # combine BERT, web, and date features
    finalFeatures = np.hstack([bertEmbed, webEncoded, dateFeat])

    # fill any NaN values with 0
    finalFeatures = np.nan_to_num(finalFeatures, nan=0.0)

    # make sure the vector has same length as what model expects
    expectedFeatures = scaler.mean_.shape[0]
    if finalFeatures.shape[1] < expectedFeatures:
        # if it's short, pad with zeros
        padding = np.zeros((1, expectedFeatures - finalFeatures.shape[1]))
        finalFeatures = np.hstack([finalFeatures, padding])
    else:
        # if too long, just trim it
        finalFeatures = finalFeatures[:, :expectedFeatures]

    # normalize using the same scaler used during training
    return scaler.transform(finalFeatures)

# final prediction function 
def predictNB(statement, web, date, models):
    # first we prepare the input vector
    inputVec = preprocessLabelInput(statement, web, date, models["web_encoder"], models["scaler"], models["bert_model"])

    # now we predict the category (like politics, health, etc.)
    category_probs = models["category_model"].predict_proba(models["category_vectorizer"].transform([statement]))[0]
    category_index = np.argmax(category_probs)
    category_label = models["category_encoder"].inverse_transform([category_index])[0]
    category_confidence = category_probs[category_index]

    # now we predict whether it's fake or true
    label_index = models["label_model"].predict(inputVec)[0]

    # calculate confidence score based on model's probability
    label_prob_true = models["label_model"].calculate_prob(1, inputVec[0])
    label_prob_fake = models["label_model"].calculate_prob(0, inputVec[0])
    total = label_prob_true + label_prob_fake

    # handle weird cases where total becomes zero or invalid
    if total == 0 or np.isnan(total) or np.isinf(total):
        label_confidence = 0.0
    else:
        label_confidence = label_prob_true / total if label_index == 1 else label_prob_fake / total

    label_text = "True" if label_index == 1 else "Fake"

    return {
        "Category": category_label,
        "Category Confidence": round(category_confidence, 4),
        "Label": label_text,
        "Label Confidence": round(label_confidence, 4)
    }

def main():
    models = load_models()

    # sample test input
    statement = "WHO praises India's Aarogya Setu app, says it helped in identifying COVID-19 clusters"
    webSource = "NDTV"
    publishDate = "Jan-21"

    # make predictions
    result = predictNB(statement, webSource, publishDate, models)

    # print the results
    print(f"Category = {result['Category']} (Confidence: {result['Category Confidence']})")
    print(f"Label = {result['Label']} (Confidence: {result['Label Confidence']})")


if __name__ == "__main__":
    main()
