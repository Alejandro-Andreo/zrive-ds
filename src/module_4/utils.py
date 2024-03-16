import logging
import pandas as pd
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent.parent.parent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_dataset() -> pd.DataFrame:
    """
    Load the dataset from the data folder.
    Returns:
        pd.DataFrame: The input dataframe containing the dataset.
    """
    loading_path = Path(DATA_PATH, "feature_frame.csv")
    df = pd.read_csv(loading_path)
    logging.info(
        f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns from {loading_path}"
    )
    return df


def push_relevant_dataframe(df: pd.DataFrame, min_products: int = 5) -> pd.DataFrame:
    """We filtered the dataframe to only include orders with at least 5 products purchased"""
    orders_size = df.groupby("order_id").outcome.sum()
    orders_of_min_size = orders_size[orders_size >= min_products].index
    return df.loc[lambda x: x.order_id.isin(orders_of_min_size)]


def load_filtered_data() -> pd.DataFrame:
    """
    Filter the dataframe to only include orders with at least 5 products purchased.
    Returns:
        pd.DataFrame: The input dataframe containing the dataset.
    """
    logging.info("Loading filtered dataset")
    return (
        load_dataset()
        .pipe(push_relevant_dataframe)
        .assign(created_at=lambda x: pd.to_datetime(x.created_at))
        .assign(order_date=lambda x: pd.to_datetime(x.order_date).dt.date)
    )
