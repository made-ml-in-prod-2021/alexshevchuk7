import os

import pytest
import pandas as pd
from typing import List
from faker import Faker
from dataclasses import dataclass
from prepare_dataset import read_data, train_target


@dataclass()
class TrainingPipelineParams:
    input_data_path: str


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
                          num_rows=10)

    dataset_fio = tmpdir.join('sample.csv')
    dataset_fio.write(sample_csv)

    return dataset_fio


@pytest.fixture()
def train_pipeline_params(dataset_path):
    params = TrainingPipelineParams
    params.input_data_path = dataset_path
    return params


@pytest.fixture()
def target_col():
    return 'target'


def test_load_dataset_using_path(train_pipeline_params, target_col):
    df = read_data(train_pipeline_params)
    assert isinstance(df, pd.DataFrame)
    assert 10 == len(df)
    assert target_col in df.keys()


def test_train_target(tmpdir, train_pipeline_params):
    df = read_data(train_pipeline_params)
    columns = ['age', 'sex', 'resteg']
    train, target = train_target(df, columns, 'target')
    assert (10, 3) == train.shape
    assert pd.Series == type(target)
    assert (10,) == target.shape