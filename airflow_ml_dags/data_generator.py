import os
from faker import Faker

import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def _generate_data(date):
    fake = Faker()
    fake.set_arguments('cp', {'min_value': 0, 'max_value': 2})
    fake.set_arguments('trestbps', {'min_value': 1, 'max_value': 150})
    fake.set_arguments('restecg', {'min_value': 1, 'max_value': 200})
    fake.set_arguments('slope', {'min_value': 0, 'max_value': 3})
    fake.set_arguments('ca', {'min_value': 0, 'max_value': 2})
    fake.set_arguments('thal', {'min_value': 0, 'max_value': 2})

    generated_csv = fake.csv(header=('cp', 'trestbps', 'restecg', 'slope', 'ca', 'thal'),
                              data_columns=(
                                            '{{pyint: cp}}',
                                            '{{pyint: trestbps}}',
                                            '{{pyint: restecg}}',
                                            '{{pyint: slope}}',
                                            '{{pyint: ca}}',
                                            '{{pyint: thal}}'
                                            ),
                              num_rows=300)

    path = os.path.join('data/raw', date)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, 'data.csv'), 'w') as fio:
        fio.write(generated_csv)


def _generate_target(date):
    fake = Faker()
    fake.set_arguments('target', {'min_value': 0, 'max_value': 1})

    generated_csv = fake.csv(header=('target',), data_columns=('{{pyint: target}}',), num_rows=300)

    path = os.path.join('data/raw', date)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, 'target.csv'), 'w') as fio:
        fio.write(generated_csv)


with DAG(
    dag_id="data_generator",
    start_date=airflow.utils.dates.days_ago(7),
    schedule_interval='@daily',
) as dag:

    generate_data = PythonOperator(
        task_id="generate_data", python_callable=_generate_data,
        op_kwargs = {'date': '{{ds}}'}
    )

    generate_target = PythonOperator(
        task_id="generate_target", python_callable=_generate_target,
        op_kwargs={'date': '{{ds}}'}
    )

    generate_data >> generate_target