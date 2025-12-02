import os
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def load_model(model_path="UI/model_paths/crop_recommendation_model.pkl"):
   
    return joblib.load(model_path)


def predict_crop(model, new_data_dict):
   
    new_df = pd.DataFrame([new_data_dict])
    prediction = model.predict(new_df)[0]
    return prediction


def recommend_top3_crops(model, input_features, crop_labels):
   
    if isinstance(input_features, dict):
        input_df = pd.DataFrame([input_features])
    else:
        input_df = input_features

    probs = model.predict_proba(input_df)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_crops = [(crop_labels[i], probs[i]) for i in top3_idx]
    return top3_crops
