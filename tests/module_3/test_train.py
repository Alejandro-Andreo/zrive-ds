import pandas as pd
import numpy as np
import pytest

from src.module_3.train import (
    evaluate_model,
    train_test_split,
    push_relevant_dataframe,
)

IMPORTANT_COLS = ["ordered_before", "abandoned_before", "global_popularity"]
TARGET_COL = "outcome"


def test_evaluate_model():
    y_test = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    good_model = evaluate_model("good_model", y_test, y_pred)
    y_pred_bad = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    bad_model = evaluate_model("bad_model", y_test, y_pred_bad)

    assert good_model > bad_model


def test_train_test_split():
    dates = pd.date_range(start="2021-01-01", end="2021-01-10")
    data = {
        "order_date": np.repeat(dates, 3),  # Each date repeated thrice
        "order_id": range(30),  # 30 unique orders
        "ordered_before": np.random.randint(0, 2, size=30),
        "abandoned_before": np.random.randint(0, 2, size=30),
        "global_popularity": np.random.rand(30),
        TARGET_COL: np.random.randint(2, size=30),  # Binary target variable
    }
    df = pd.DataFrame(data)

    train_size = 0.7  # 70% of data for training

    X_train, X_val, y_train, y_val = train_test_split(df, train_size)

    assert all(
        col in X_train.columns for col in IMPORTANT_COLS
    ), "Training features missing important columns."
    assert all(
        col in X_val.columns for col in IMPORTANT_COLS
    ), "Validation features missing important columns."
    assert y_train.name == TARGET_COL, "Training target column incorrect."
    assert y_val.name == TARGET_COL, "Validation target column incorrect."

    total_size = df.shape[0]
    expected_train_size = int(total_size * train_size)
    assert (
        len(X_train) == expected_train_size
    ), "Training set size does not match expected proportion."


@pytest.fixture
def mock_dataframe():
    """Create a mock DataFrame to simulate order data."""
    data = {
        "order_id": [1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6],
        "product_id": range(11),
        "outcome": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    return pd.DataFrame(data)


def test_push_relevant_dataframe(mock_dataframe):
    actual = push_relevant_dataframe(mock_dataframe, 3).reset_index(drop=True)
    expected = pd.DataFrame(
        {"order_id": [2, 2, 2], "product_id": [2, 3, 4], "outcome": [1, 1, 1]}
    )

    pd.testing.assert_frame_equal(actual, expected)
