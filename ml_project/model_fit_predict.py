import pickle
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score

from entities.cv_params import CVParams
from entities.feature_params import FeatureParams
from entities.train_params import TrainingParams
from prepare_dataset import read_data, train_target


Estimator = Union[GaussianNB, RandomForestClassifier, LogisticRegression]


def get_stacking_feature(estimator: Estimator,
                         df: pd.DataFrame,
                         target: pd.Series,
                         cv_params: CVParams) -> np.array:

    kf = KFold(n_splits=cv_params.folds, shuffle=True, random_state=cv_params.random_state)
    response = np.zeros(df.shape[0])

    for train_index, val_index in kf.split(df):
        train = df.iloc[train_index]
        y_train = target.iloc[train_index]
        val = df.iloc[val_index]

        estimator.fit(train, y_train)
        y_preds = estimator.predict_proba(val)[:, 1]
        response[val_index] = y_preds

    return response


def merge_features(features: Tuple) -> pd.DataFrame:
    return pd.DataFrame(np.c_[features])


def train_final_model(stacked_features: pd.DataFrame,
                      target: pd.Series,
                      train_params: TrainingParams) -> Estimator:

    estimator = train_params.final_estimator

    if estimator == 'RandomForestClassifier':
        model = RandomForestClassifier()
    elif estimator == 'LogisticRegression':
        model = LogisticRegression(solver='newton-cg')
    elif estimator == 'GaussianNB':
        model = GaussianNB()
    else:
        raise NotImplementedError()

    model.fit(stacked_features, target)

    return model


def get_prediction(model: object,
                   stacked_data: pd.DataFrame,
                   ) -> np.ndarray:

    predictions = model.predict(stacked_data)
    return predictions


def cv_score(train_df: pd.DataFrame,
             target: pd.Series,
             train_params: TrainingParams,
             cv_params: CVParams) -> Tuple[float, float, float]:

    """
        Calculates accuracy_score, recall_score and precision_score for k-fold cross-validation
        with shuffling.
        Arguments:
            train_df: pd.Dataframe (train data)
            target: pd.Series (class labels)
            train_params: TrainingParams (final estimator)
            cv_params: CVParams (cross-validation parameters)

        Returns: tuple:
            mean_accruacy
            mean_recall
            mean_precision
    """
    kf = KFold(n_splits=cv_params.folds, shuffle=True, random_state=cv_params.random_state)
    final_estimator = train_params.final_estimator
    if final_estimator == 'RandomForestClassifier':
        estimator = RandomForestClassifier()
    elif final_estimator == 'LogisticRegression':
        estimator = LogisticRegression()
    elif final_estimator == 'GaussianNB':
        estimator = GaussianNB()
    else:
        raise NotImplementedError()

    accuracies = []
    recalls = []
    precisions = []

    for train_index, val_index in kf.split(train_df):
        train = train_df.iloc[train_index]
        y_train = target.iloc[train_index]
        val = train_df.iloc[val_index]
        y_val = target.iloc[val_index]

        estimator.fit(train, y_train)
        y_preds = estimator.predict(val)
        accuracies.append(accuracy_score(y_val, y_preds))
        recalls.append(recall_score(y_val, y_preds))
        precisions.append(precision_score(y_val, y_preds))

        mean_accuracy = np.array(accuracies).mean()
        mean_recall = np.array(recalls).mean()
        mean_precision = np.array(precisions).mean()

    return mean_accuracy, mean_recall, mean_precision


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output