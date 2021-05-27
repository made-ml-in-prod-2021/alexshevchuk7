import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from entities.feature_params import FeatureParams


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:

    if isinstance(categorical_df, pd.Series):
        categorical_df = np.array(categorical_df).reshape(-1, 1)

    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            ('ohe', OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:

    if isinstance(numerical_df, pd.Series):
        numerical_df = np.array(numerical_df).reshape(-1, 1)

    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
         ('minmax', MinMaxScaler()),
         ]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.fit_transform(df))


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                list(set(params.categorical_features).intersection(set(params.selected_features))),
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                list(set(params.numerical_features).intersection(set(params.selected_features))),
            ),
        ]
    )
    return transformer