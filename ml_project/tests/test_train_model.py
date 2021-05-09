import pytest
import pandas as pd
import numpy as np
import yaml
from dataclasses import dataclass
from faker import Faker
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from train_model import (train_configure,
                         score_model,
                         stacking_features_for_inference,
                         predict_from_csv,
                         )
from prepare_dataset import read_data, train_target

@dataclass()
class TrainingPipelineParams:
    input_data_path: str


@pytest.fixture()
def config_path():
    return 'configs/train_config.yml'


@pytest.fixture()
def path_to_data():
    return 'data/heart.csv'


@pytest.fixture()
def path_to_model():
    return 'models/model.pkl'


@pytest.fixture()
def evaluation_dataset_path(tmpdir):
    fake = Faker()
    fake.set_arguments('age', {'min_value': 29, 'max_value': 100})
    fake.set_arguments('sex', {'min_value': 0, 'max_value': 1})
    fake.set_arguments('cp', {'min_value': 0, 'max_value': 2})
    fake.set_arguments('trestbps', {'min_value': 1, 'max_value': 150})
    fake.set_arguments('chol', {'min_value': 1, 'max_value': 100})
    fake.set_arguments('fbs', {'min_value': 0, 'max_value': 1})
    fake.set_arguments('restecg', {'min_value': 1, 'max_value': 200})
    fake.set_arguments('thalach', {'min_value': 0, 'max_value': 2})
    fake.set_arguments('exang', {'min_value': 0, 'max_value': 1})
    fake.set_arguments('oldpeak', {'min_value': 0, 'max_value': 2})
    fake.set_arguments('slope', {'min_value': 0, 'max_value': 3})
    fake.set_arguments('ca', {'min_value': 0, 'max_value': 2})
    fake.set_arguments('thal', {'min_value': 0, 'max_value': 2})

    evaluation_csv = fake.csv(header=('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'),
                          data_columns=('{{pyint: age}}',
                                        '{{pyint: sex}}',
                                        '{{pyint: cp}}',
                                        '{{pyint: trestbps}}',
                                        '{{pyint: chol}}',
                                        '{{pyint: fbs}}',
                                        '{{pyint: restecg}}',
                                        '{{pyint: thalach}}',
                                        '{{pyint: exang}}',
                                        '{{pyint: oldpeak}}',
                                        '{{pyint: slope}}',
                                        '{{pyint: ca}}',
                                        '{{pyint: thal}}'
                                        ),
                          num_rows=10)

    dataset_fio = tmpdir.join('evaluation.csv')
    dataset_fio.write(evaluation_csv)

    return dataset_fio


@pytest.fixture()
def model_params(config_path):
    params = train_configure(config_path)
    return params


def test_train_configure(config_path):
    params = train_configure(config_path)
    assert 'LogisticRegression' == params.train_params.final_estimator
    assert 'data/heart.csv' == params.input_data_path
    assert 'trestbps' in params.feature_params.selected_features
    assert 9 == params.cv_params.folds


def test_score_model(config_path):
    path, scores = score_model(config_path)
    assert 1 >= max(scores)
    assert 0 <= min(scores)
    assert False == np.any(np.isnan(scores))


def test_stacking_features_inference(model_params, evaluation_dataset_path):
    stacked_features = stacking_features_for_inference(model_params, evaluation_dataset_path)
    assert (10, 3) == stacked_features.shape


def test_make_predictions(config_path, path_to_model, evaluation_dataset_path):
    predictions = predict_from_csv(config_path, path_to_model, evaluation_dataset_path)
    assert (10,) == predictions.shape
    assert len(set(predictions)) <= 2
    assert 1 in set(predictions) or 0 in set(predictions)