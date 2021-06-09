import os
import json
from typing import Union, Dict, Tuple
import pandas as pd
import numpy as np
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score

import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.email import send_email

Estimator = Union[GaussianNB, RandomForestClassifier, LogisticRegression]
estimators = {'RandomForestClassifier': RandomForestClassifier,
              'LogisticRegression': LogisticRegression,
              'GaussianNB': GaussianNB}
metrics = {'accuracy': accuracy_score,
           'recall': recall_score,
           'precision': precision_score,
           }
PATH_DATA = 'data/raw'
PATH_PROCESSED = 'data/processed'
PATH_MODEL = 'data/models'
PATHS = [PATH_DATA, PATH_PROCESSED, PATH_MODEL]
PREMODELS = ['premodel_1.pkl', 'premodel_2.pkl', 'premodel_3.pkl']
TRAIN_VAL_SPLIT = 0.8


def failure_email(context):
    email_title = "Airflow Task {task_id} Failed".format(context['task_instance'].task_id)
    email_body = "{task_id} in {dag_id} failed.".format(context['task_instance'].task_id, context['task_instance'].dag_id)
    send_email('shevchuk.alex@gmail.com', email_title, email_body)


def get_stacking_feature(estimator: Estimator,
                         df: pd.DataFrame,
                         target: pd.Series,
                         ) -> np.array:

    kf = KFold(n_splits=7, shuffle=True, random_state=0)
    response = np.zeros(df.shape[0])

    for train_index, val_index in kf.split(df):
        train = df.iloc[train_index]
        y_train = target.iloc[train_index].values.ravel()
        val = df.iloc[val_index]

        estimator.fit(train, y_train)
        y_preds = estimator.predict_proba(val)[:, 1]
        response[val_index] = y_preds

    return response


def merge_features(features: Tuple) -> pd.DataFrame:
    return pd.DataFrame(np.c_[features])


def _process_data(date):

    data_path = os.path.join(PATH_DATA, date, 'data.csv')
    train = pd.read_csv(data_path)
    target_path = os.path.join(PATH_DATA, date, 'target.csv')
    target = pd.read_csv(target_path)

    estimators = [GaussianNB(), RandomForestClassifier(), LogisticRegression(solver='newton-cg')]
    features = [get_stacking_feature(estimator, train, target)
                for estimator in estimators]

    stacking_df = merge_features(tuple(features))

    path = os.path.join(PATH_PROCESSED, date)
    if not os.path.exists(path):
        os.makedirs(path)

    stacking_df.to_csv(os.path.join(path, 'processed_data.csv'), index=False)


def _train_val_split(date):
    path = os.path.join(PATH_PROCESSED, date, 'processed_data.csv')
    data = pd.read_csv(path)

    data_len = len(data)
    train_set = data[: int(data_len * TRAIN_VAL_SPLIT)]
    val_set = data[int(data_len * TRAIN_VAL_SPLIT) : ]

    train_set.to_csv(os.path.join(PATH_PROCESSED, date, 'train.csv'), index=False)
    val_set.to_csv(os.path.join(PATH_PROCESSED, date, 'val.csv'), index=False)


def _get_inference_models(date):
    data_path = os.path.join(PATH_DATA, date, 'data.csv')
    train = pd.read_csv(data_path)
    target_path = os.path.join(PATH_DATA, date, 'target.csv')
    target = pd.read_csv(target_path)

    path = os.path.join(PATH_MODEL, date)
    if not os.path.exists(path):
        os.makedirs(path)

    estimators = [GaussianNB(), RandomForestClassifier(), LogisticRegression(solver='newton-cg')]

    for estimator, model_name in zip(estimators, PREMODELS):
        model = estimator.fit(train, target)

        with open(os.path.join(PATH_MODEL, date, model_name), "wb") as f:
            pickle.dump(model, f)


def _get_final_model(date):

    train_data = pd.read_csv(os.path.join(PATH_PROCESSED, date, 'train.csv'))
    target = pd.read_csv(os.path.join(PATH_DATA, date, 'target.csv'))
    y_train = target[: int(len(target) * TRAIN_VAL_SPLIT)]

    final_model = LogisticRegression(solver='newton-cg', max_iter=100)
    final_model.fit(train_data, y_train)

    path = os.path.join(PATH_MODEL, date)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(PATH_MODEL, date, 'model.pkl'), "wb") as f:
        pickle.dump(final_model, f)


def _get_val_metrics(date):

    val_data = pd.read_csv(os.path.join(PATH_PROCESSED, date, 'val.csv'))
    target = pd.read_csv(os.path.join(PATH_DATA, date, 'target.csv'))
    y_val = target[int(len(target) * TRAIN_VAL_SPLIT) : ]

    with open(os.path.join(PATH_MODEL, date, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    y_preds = model.predict(val_data)

    metrics_results = {name: [] for name in metrics.keys()}

    for score in metrics.keys():
        metrics_results[score].append(metrics[score](y_val, y_preds))

    with open(os.path.join(PATH_MODEL, date, 'scores.json'), "w") as metric_file:
        json.dump(metrics_results, metric_file)


with DAG(
    dag_id="get_model",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval='@weekly',
    on_failure_callback=failure_email,
) as dag:

    file_names = ['data.csv', 'processed_data.csv', 'model.pkl']
    paths = [os.path.join('home/alexey', folder_name, '{{ds}}', file_name)
             for folder_name, file_name in zip(PATHS, file_names)]

    path_sensors = [FileSensor(task_id=f'sensor_{str(sensor_id)}',
                               poke_interval=2,
                               timeout=2 * 5,
                               mode='poke',
                               filepath=f'{path_to_file}',
                               )
                    for sensor_id, path_to_file in enumerate(paths)]

    process_data = PythonOperator(
        task_id="process_data", python_callable=_process_data,
        op_kwargs = {'date': '{{ds}}'}
    )

    train_validation_split = PythonOperator(
        task_id="generate_target", python_callable=_train_val_split,
        op_kwargs={'date': '{{ds}}'}
    )

    get_model = PythonOperator(
        task_id="get_model", python_callable=_get_final_model,
        op_kwargs={'date': '{{ds}}'}
    )

    inference_models = PythonOperator(
        task_id="inference_models", python_callable=_get_inference_models,
        op_kwargs={'date': '{{ds}}'}
    )

    get_metrics = PythonOperator(
        task_id="get_metrics", python_callable=_get_val_metrics,
        op_kwargs={'date': '{{ds}}'}
    )

    process_data >> path_sensors[0] >> train_validation_split >> path_sensors[1] >> get_model
    get_model >> path_sensors[2] >> inference_models >> get_metrics