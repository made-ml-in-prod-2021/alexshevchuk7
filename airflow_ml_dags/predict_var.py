import os
from typing import Union, Dict, Tuple
import pandas as pd
import numpy as np
import pickle

import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.email import send_email


PATH_DATA = 'data/raw'
PATH_MODEL = '{{var.value.model_path}}'
PATH_PREDICTIONS = 'data/predictions'
PREMODELS = ['premodel_1.pkl', 'premodel_2.pkl', 'premodel_3.pkl']


def failure_email(context):
    email_title = "Airflow Task {task_id} Failed".format(context['task_instance'].task_id)
    email_body = "{task_id} in {dag_id} failed.".format(context['task_instance'].task_id, context['task_instance'].dag_id)
    send_email('shevchuk.alex@gmail.com', email_title, email_body)


def get_features(model, data):
    return model.predict(data)


def merge_features(features: Tuple) -> pd.DataFrame:
    return pd.DataFrame(np.c_[features])


def _predict(date):

    data_path = os.path.join(PATH_DATA, date, 'data.csv')
    data = pd.read_csv(data_path)

    if os.path.exists(PATH_MODEL):
        model_dir = list(os.scandir(PATH_MODEL))[-1].name

        features = []

        for model_name in PREMODELS:
            with open(os.path.join(PATH_MODEL, model_dir, model_name), 'rb') as f:
                premodel = pickle.load(f)

            feature = get_features(premodel, data)
            features.append(feature)

        stacking_df = merge_features(tuple(features))

        with open(os.path.join(PATH_MODEL, model_dir, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict(stacking_df)

        predictions = pd.DataFrame(predictions)

        path = os.path.join(PATH_PREDICTIONS, date)
        if not os.path.exists(path):
            os.makedirs(path)

        predictions.to_csv(os.path.join(PATH_PREDICTIONS, date, 'predictions.csv'))


with DAG(
        dag_id="predict_var",
        start_date=airflow.utils.dates.days_ago(1),
        schedule_interval='@daily',
        on_failure_callback=failure_email,
) as dag:

    predict = PythonOperator(
        task_id="process_data", python_callable=_predict,
        op_kwargs={'date': '{{ds}}'}
    )

    predict