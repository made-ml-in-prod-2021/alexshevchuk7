import pandas as pd
from typing import Tuple

from entities.train_pipeline_params import TrainingPipelineParams

def read_data(params: TrainingPipelineParams) -> pd.DataFrame:
    print(params)
    path = params.input_data_path
    df = pd.read_csv(path)
    return df


def train_target(data: pd.DataFrame, columns: list, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    train = data[columns]
    y = data[target]
    return train, y