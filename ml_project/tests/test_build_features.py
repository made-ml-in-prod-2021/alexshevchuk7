import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
from build_features import (process_categorical_features,
                            process_numerical_features,
                            build_transformer,
                            make_features)


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    selected_features: List[str]
    target_col: str


@pytest.fixture()
def params():
    params = FeatureParams
    params.categorical_features = ['cp', 'sex']
    params.numerical_features = ['age', 'chol']
    params.selected_features = ['age', 'cp']

    return params


@pytest.fixture()
def df():
    df = pd.DataFrame(index=range(3), data={'age': [10, np.nan, 20], 'cp': [1, 2, 3]})
    return df


def test_process_categorical_features(df):
    assert isinstance(process_categorical_features(df['cp']), pd.DataFrame)
    assert (3, 3) == process_categorical_features(df['cp']).shape


def test_process_numerical_features(df):
    assert isinstance(process_numerical_features(df['age']), pd.DataFrame)
    assert (3, 1) == process_numerical_features(df['age']).shape
    assert 0.5 == process_numerical_features(df['age']).iloc[1, 0]


def test_make_features(params, df):
    transformer = build_transformer(params)
    assert isinstance(make_features(transformer, df), pd.DataFrame)
    assert (3, 4) == make_features(transformer, df).shape
    assert np.allclose(np.array([1, 1.5, 2]), make_features(transformer, df).values.sum(axis=1))
