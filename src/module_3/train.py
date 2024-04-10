import logging
import pandas as pd
import pickle

from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

C_TUNING = [1e-8, 1e-6, 1e-4, 1e-2, 1]
IMPORTANT_COLS = ["ordered_before", "abandoned_before", "global_popularity"]

TARGET_COL = "outcome"

HOLDOUT = 0.8

DATA_PATH = "/home/alejandro/Zrive"


def evaluate_model(model_name: str, y_test: pd.Series, y_pred: pd.Series) -> float:
    """
    Evaluate the model using precision-recall AUC and ROC AUC

    Args:
        model_name (str): The name of the model being evaluated.
        y_test (pd.Series): The true labels of the test set.
        y_pred (pd.Series): The predicted labels of the test set.

    Returns:
        float: The precision-recall AUC score of the model.
    """
    precision_, recall_, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall_, precision_)
    roc_auc = roc_auc_score(y_test, y_pred)
    logging.info(f"{model_name} - PR AUC: {pr_auc:.2f}, ROC AUC: {roc_auc:.2f}")
    return pr_auc


def feature_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[IMPORTANT_COLS], df[TARGET_COL]


def train_test_split(
    df: pd.date_range, train_size: float
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the dataset into training and validation sets based on a given train size since there is a time component.

    Args:
        df (pd.date_range): The input dataframe containing the dataset.
        train_size (float): The proportion of the dataset to be used for training.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Dataframe, pd.Series]: A tuple containing the training and validation sets.

    """
    daily_orders = df.groupby("order_date").order_id.nunique()
    cumsum_daily_orders = daily_orders.cumsum() / daily_orders.sum()
    cutoff = cumsum_daily_orders[cumsum_daily_orders <= train_size].idxmax()

    X_train, y_train = feature_split(df[df.order_date <= cutoff])
    X_val, y_val = feature_split(df[df.order_date > cutoff])

    logging.info(
        f"Cutoff is {cutoff}. Train size is {X_train.shape[0]} and validation size is {X_val.shape[0]}"
    )
    return X_train, X_val, y_train, y_val


def save_model(model: BaseEstimator, model_name: str) -> None:
    """
    Save the model to a file

    Args:
        model (BaseEstimator): The model to be saved.
        model_name (str): The name of the model to be saved.
    """
    logging.info(f"Saving model to {model_name}.pkl")

    pickle.dump(model, open(Path(__file__).parent / f"{model_name}.pkl", "wb"))


def model_selection(df: pd.DataFrame) -> None:
    """
    Train and evaluate the model using different values of C.

    Args:
        df (pd.DataFrame): The input dataframe containing the dataset.
    """
    X_train, X_val, y_train, y_val = train_test_split(df, HOLDOUT)
    best_auc = 0
    for c in C_TUNING:
        logging.info(f"Training model with C={c}")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c))
        model.fit(X_train, y_train)
        pr_auc = evaluate_model(
            f"Logistic Regression (C={c})", y_val, model.predict_proba(X_val)[:, 1]
        )
        if pr_auc > best_auc:
            logging.info(f"New best model found with C={c}")
            best_auc = pr_auc
            best_c = c
    logging.info(f"Best model found with C={best_c}")
    best_model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c))
    X, y = feature_split(df)
    best_model.fit(X, y)
    save_model(best_model, f"best_model_{best_c}")


def push_relevant_dataframe(df: pd.DataFrame, min_products: int = 5) -> pd.DataFrame:
    """We filtered the dataframe to only include orders with at least 5 products purchased"""
    orders_size = df.groupby("order_id").outcome.sum()
    orders_of_min_size = orders_size[orders_size >= min_products].index
    return df.loc[lambda x: x.order_id.isin(orders_of_min_size)]


def load_dataset() -> pd.DataFrame:
    """
    Load the dataset from the data folder.

    Returns:
        pd.DataFrame: The input dataframe containing the dataset.
    """
    df = pd.read_csv(Path(DATA_PATH, "feature_frame.csv"))
    return (
        df.pipe(push_relevant_dataframe)
        .assign(created_at=lambda x: pd.to_datetime(x.created_at))
        .assign(order_date=lambda x: pd.to_datetime(x.order_date).dt.date)
    )


def main() -> None:
    feature_frame = load_dataset()
    model_selection(feature_frame)


if __name__ == "__main__":
    main()
