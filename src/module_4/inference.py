import os
import pandas as pd
import json
import pickle
from .utils import load_filtered_data
from push_model import Push_Model


def load_data(data: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(data)
    return load_filtered_data(df)


def load_model(model_path: str) -> Push_Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = pickle.load(open(model_path, "rb"))
    return model


def handler_predict(event: dict, _) -> dict:
    """
    Receives a json with a field "users" that contains:
        "users": {
            "user_id": {
                "feature 1": feature value,
                "feature 2": feature value, ...
                },
            "user_id2": {
                "feature 1": feature value,
                "feature 2": feature value, ...
                }.
        }
        "model_path" : value
    Output should be a json with a field "body" with fields "prediction":
        {
            {"user_id": prediction,
            "user_id2": prediction ...}
        }
    """
    data_to_predict = load_filtered_data(event["users"])
    model_path = event.get("model_path")
    model = load_model(model_path)
    prediction = model.predict(data_to_predict)
    return {
        "statusCode": "200",
        "body": json.dumps({"prediction": prediction.to_dict()}),
    }
