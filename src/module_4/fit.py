import logging
import pickle
import json
from datetime import datetime
from pathlib import Path
from sklearn.base import BaseEstimator
from push_model import Push_Model
from utils import load_filtered_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_model(model: BaseEstimator, model_name: str) -> str:
    """
    Save the model to the models folder.

    Args:
        model (BaseEstimator): The trained model.
        model_name (str): The name of the model.

    Returns:
        str: The path to the saved model.
    """
    model_path = Path(Path(__file__).parent, "models", f"{model_name}.pkl")
    pickle.dump(model, open(model_path, "wb"))
    logging.info(f"Saved model to {model_path}")
    return str(model_path)


def create_model_name() -> str:
    """
    Create a model name from the event dictionary

    Args:
        event (dict): The event dictionary.

    Returns:
        str: The model name.
    """
    train_date = datetime.today().strftime("%Y-%m-%d")
    model_name = "push_" + train_date
    return model_name


def extract_params(event: dict) -> dict:
    model_parametrisation = event["model_parametrisation"]
    classifier_parametrisation = model_parametrisation["classifier_parametrisation"]
    return classifier_parametrisation


def handler_fit(event: dict, _) -> dict:
    """
    Args:
        event (dict): The event dictionary.
         Example:
        {
            "model_parametrisation":
            {   "classifier_parametrisation":
                {
                    "max_iter"=100,
                    "learning_rate"=0.05,
                    "max_depth"=10,
                    "random_state"=0
                }
            }
        }
    Return: Dict
    """
    model_parametrisation = extract_params(event)
    df = load_filtered_data()
    model = Push_Model(model_parametrisation)
    model.fit(df)
    model_name = create_model_name()
    model_path = save_model(model, model_name)
    return {"statusCode": "200", "body": json.dumps({"model_path": model_path})}
