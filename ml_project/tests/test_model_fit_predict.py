import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass
from faker import Faker
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score

from model_fit_predict import (get_stacking_feature,
                               merge_features,
                               train_final_model,
                               get_prediction,
                               cv_score)
from prepare_dataset import read_data, train_target

@dataclass()
class TrainingPipelineParams:
    input_data_path: str

@dataclass()
class CVParams:
    folds: int
    random_state: int

@dataclass()
class TrainingParams:
    final_estimator: List[str]
    preprocessing: List[str]

@dataclass()
class FeatureParams:
    selected_features: List[str]
    target_col: str


@pytest.fixture()
def cv_params():
    params = CVParams
    params.folds = 9
    params.random_state = 0
    return params


@pytest.fixture()
def train_params():
    params = TrainingParams
    params.final_estimator = 'LogisticRegression'
    params.preprocessing = None
    return params


@pytest.fixture()
def feature_params():
    params = FeatureParams
    params.selected_features = ['age', 'resteg']
    params.target_col = 'target'
    return params
  
  
@pytest.fixture()
def metrics():
    metrics = {'accuracy': accuracy_score,
               'recall': recall_score,
               'precision': precision_score,
               }
    return metrics


@pytest.fixture()
def estimator():
    return RandomForestClassifier()


@pytest.fixture()
def final_estimator():
    return LogisticRegression()


@pytest.fixture()
def dataset_path(tmpdir):
    fake = Faker()
    fake.set_arguments('age', {'min_value': 29, 'max_value': 100})
    fake.set_arguments('sex', {'min_value': 0, 'max_value': 1})
    fake.set_arguments('resteg', {'min_value': 0, 'max_value': 3})
    fake.set_arguments('target', {'min_value': 0, 'max_value': 1})

    sample_csv = fake.csv(header=('age', 'sex', 'resteg', 'target'),
                          data_columns=('{{pyint: age}}',
                                        '{{pyint: sex}}',
                                        '{{pyint: resteg}}',
                                        '{{pyint: target}}'),
                          num_rows=100)

    dataset_fio = tmpdir.join('sample.csv')
    dataset_fio.write(sample_csv)

    return dataset_fio


@pytest.fixture()
def train_pipeline_params(dataset_path):
    params = TrainingPipelineParams
    params.input_data_path = dataset_path
    return params


@pytest.fixture()
def stacked_data(tmpdir):
    fake = Faker()
    fake.set_arguments('0', {'min_value': 0, 'max_value': 1})
    fake.set_arguments('1', {'min_value': 0, 'max_value': 1})

    dataset = fake.csv(header=('0', '1'),
                          data_columns=('{{pyint: 0}}',
                                        '{{pyint: 1}}',
                                        ),
                          num_rows=10)

    dataset_fio = tmpdir.join('stacked_data.csv')
    dataset_fio.write(dataset)

    return dataset_fio

def test_get_stacking_feature(estimator, train_pipeline_params, cv_params):
    df = read_data(train_pipeline_params)
    train_df, target = train_target(df, ['age', 'resteg'], 'target')
    assert 100 == get_stacking_feature(estimator, train_df, target, cv_params).shape[0]
    assert 1 >= get_stacking_feature(estimator, train_df, target, cv_params).max()
    assert 0 <= get_stacking_feature(estimator, train_df, target, cv_params).max()


def test_merge_features(estimator, final_estimator, train_pipeline_params, cv_params):
    df = read_data(train_pipeline_params)
    train_df, target = train_target(df, ['age', 'resteg'], 'target')
    x1 = get_stacking_feature(estimator, train_df, target, cv_params)
    x2 = get_stacking_feature(final_estimator, train_df, target, cv_params)
    assert (100, 2) == merge_features((x1, x2)).shape


def test_cv_score(estimator, final_estimator, train_pipeline_params, metrics, cv_params ,train_params):
    df = read_data(train_pipeline_params)
    train_df, target = train_target(df, ['age', 'resteg'], 'target')
    x1 = get_stacking_feature(estimator, train_df, target, cv_params)
    x2 = get_stacking_feature(final_estimator, train_df, target, cv_params)
    stacked_features = merge_features((x1, x2))
    mean_scores = cv_score(stacked_features, target, metrics, train_params, cv_params)
    assert 0 <= mean_scores['mean_accuracy'] <= 1
    assert 0 <= mean_scores['mean_recall'] <= 1
    assert 0 <= mean_scores['mean_precision'] <= 1


@pytest.fixture()
def model(estimator, final_estimator, train_pipeline_params, cv_params ,train_params):
    df = read_data(train_pipeline_params)
    train_df, target = train_target(df, ['age', 'resteg'], 'target')
    x1 = get_stacking_feature(estimator, train_df, target, cv_params)
    x2 = get_stacking_feature(final_estimator, train_df, target, cv_params)
    stacked_features = merge_features((x1, x2))
    model = train_final_model(stacked_features, target, train_params)
    return model


def test_get_predictions(model, stacked_data):
    stacked_df = pd.read_csv(stacked_data)
    predictions = get_prediction(model, stacked_df)
    assert (10,) == predictions.shape
    assert len(set(predictions)) <= 2
    assert 1 in set(predictions) or 0 in set(predictions)
