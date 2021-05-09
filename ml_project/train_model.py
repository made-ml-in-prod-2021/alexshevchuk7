import json
import logging
import os
import sys
import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score

from prepare_dataset import read_data, train_target
from entities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from build_features import make_features, build_transformer
from model_fit_predict import (get_stacking_feature,
                               merge_features,
                               train_final_model,
                               get_prediction,
                               cv_score,
                               serialize_model,
                               )


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_configure(config_path: str):
    training_params = read_training_pipeline_params(config_path)
    return training_params


def get_data(params):
    selected_features = params.feature_params.selected_features
    target = params.feature_params.target_col
    data = read_data(params)
    train, target = train_target(data, selected_features, target)

    return train, target


def process_data(train, target, params):
    if params.train_params.preprocessing != 'None':
        transformer = build_transformer(params.feature_params)
        train = make_features(transformer, train)
        return train, target
    else:
        return train, target


def get_stacked_df(train, target, params):

    estimators = [GaussianNB(), RandomForestClassifier(), LogisticRegression(solver='newton-cg')]
    features = [get_stacking_feature(estimator, train, target, params.cv_params)
                for estimator in estimators]

    stacking_df = merge_features(tuple(features))

    return stacking_df


def stacking_features_for_inference(params, evaluation_dataset_path):
    selected_features = params.feature_params.selected_features
    evaluation_dataset = pd.read_csv(evaluation_dataset_path)
    train, target = get_data(params)
    train_len = len(train)

    train = train.append(evaluation_dataset[selected_features])
    combined_df, target = process_data(train, target, params)

    train = combined_df[:train_len]
    inference_data = combined_df[train_len:]

    estimators = [GaussianNB(), RandomForestClassifier(), LogisticRegression(solver='newton-cg')]
    inference_data.to_csv('inference.csv')
    features = []
    for estimator in estimators:
        estimator.fit(train, target)
        predictions = estimator.predict_proba(inference_data)[:, 1]
        features.append(predictions)

    stacked_features = merge_features(tuple(features))

    return stacked_features


def score_model(path_to_params):
    training_parameters = train_configure(path_to_params)
    logger.info(f'start training with params {training_parameters}')
    train, target = get_data(training_parameters)
    train, target = process_data(train, target, training_parameters)
    logger.info(f'data.shape is {train.shape}')

    stacking_df = get_stacked_df(train, target, training_parameters)

    scores = cv_score(stacking_df, target, training_parameters.train_params, training_parameters.cv_params)
    logger.info(f'Scores were evaluated using {training_parameters.cv_params.folds}-fold cross-validation')

    with open(training_parameters.metric_path, "w") as metric_file:
        json.dump(scores, metric_file)
    logger.info(f'Mean CV accuracy: {scores[0]}')
    logger.info(f'mean CV recall: {scores[1]}')
    logger.info(f'mean CV precision: {scores[2]}')

    logger.info(f'Serializing final model to {training_parameters.output_model_path}...')
    final_model = train_final_model(stacking_df, target, training_parameters.train_params)
    path_to_model = serialize_model(final_model, training_parameters.output_model_path)
    return path_to_model, scores


def predict_from_csv(config_path: str, path_to_model: str, evaluation_data_path: str):
    training_parameters = train_configure(config_path)
    logger.info(f'reading data...')
    try:
        train, target = get_data(training_parameters)
        train, target = process_data(train, target, training_parameters)
    except FileNotFoundError:
        logger.info(f'CSV file was not found')
        raise NotImplementedError()

    logger.info(f'data.shape is {train.shape}')

    stacked_df = stacking_features_for_inference(training_parameters, evaluation_data_path)

    try:
        with open(path_to_model, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        logger.info(f'model was not found')
        raise NotImplementedError()

    predictions = get_prediction(model, stacked_df)
    logger.info(f'Predictions: {predictions}')

    return predictions


@click.group()
def cli():
    pass


@cli.command(name="score")
@click.argument("config_path")
def train_command(config_path: str):
    score_model(config_path)


@cli.command(name="predict")
@click.argument("config_path")
@click.argument("path_to_model")
@click.argument("data_path")
def predict(config_path: str, path_to_model: str, data_path: str):
    predict_from_csv(config_path, path_to_model, data_path)


if __name__ == "__main__":
    cli()
