import os
import pickle
from typing import List, Union, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def load_object(path: str) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


class PredictionModel(BaseModel):
    data: List


class Response(BaseModel):
    prediction: int


model: Optional[object] = None
partial_models: List[object] = []


def make_predict(
    data: List, partial_models, model: object,
) -> List[Response]:

    features = []
    for estimator in partial_models:
        predictions = estimator.predict_proba(data)[:, 1]
        features.append(predictions)

    stacked_features = np.c_[tuple(features)]
    predicts = model.predict(stacked_features)

    return predicts


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_models():
    global model, partial_models
    partial_models_path = ['partial_gauss.pkl', 'partial_rfc.pkl', 'partial_lrg.pkl']

    partial_models = []
    
    for model_path in partial_models_path:
        partial_models.append(load_object(model_path))

    model_path = 'model.pkl'
    model = load_object(model_path)


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=int)
def predict(request: PredictionModel):
    if len(request.data[0]) != 6:
        raise HTTPException(status_code=400, detail="Expected sample length shall be 6")

    if sum([(isinstance(data_item, int) or isinstance(data_item, int)) for data_item in request.data[0]]) != 6:
        raise HTTPException(status_code=400, detail="Expect only floats or ints in input data")

    return make_predict(request.data, partial_models, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
