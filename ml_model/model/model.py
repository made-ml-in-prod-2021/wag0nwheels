import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ml_model.enities.training_params import TrainingParams
from ml_model.enities.features_params import FeatureParams


def train(data: pd.DataFrame, target: pd.Series, use_log_trick: FeatureParams,
          training_params: TrainingParams):
    if training_params.train_model == 'knn':
        if training_params.model_params:
            model = KNeighborsClassifier(**training_params.model_params)
        else:
            model = KNeighborsClassifier()
    elif training_params.train_model == 'rfc':
        if training_params.model_params:
            model = RandomForestClassifier(**training_params.model_params)
        else:
            model = RandomForestClassifier()
    else:
        model = LogisticRegression()
    if use_log_trick.use_log_trick:
        model.fit(data, np.log1p(target))
    else:
        model.fit(data, target)
    return model


def predict(model, features, use_log_trick: bool = False):
    if use_log_trick:
        predicts = np.expm1(model.predict(features))
    else:
        predicts = model.predict(features)
    return predicts


def evaluate(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts),
    }


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
