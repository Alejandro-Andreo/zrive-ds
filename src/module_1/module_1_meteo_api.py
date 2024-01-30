import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import time
import logging
from requests.models import Response


API_URL = "https://climate-api.open-meteo.com/v1/climate?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_10cm_mean"]
MODELS = [
    "CMCC_CM2_VHR4",
    "FGOALS_f3_H",
    "HiRAM_SIT_HR",
    "MRI_AGCM3_2_S",
    "EC_Earth3P_HR",
    "MPI_ESM1_2_XR",
    "NICAM16_8S",
]
MAX_CALL_ATTEMPTS = 50

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_response(response: Response) -> bool:
    if not isinstance(response, dict):
        return False

    # Check if the required keys are in the response
    required_keys = [
        "latitude",
        "longitude",
        "generationtime_ms",
        "timezone",
        "timezone_abbreviation",
        "daily",
        "daily_units",
    ]
    if not all(key in response for key in required_keys):
        return False

    # Check if 'daily' is a dictionary
    if not isinstance(response["daily"], dict):
        return False

    # Check if 'daily_units' is a dictionary
    if not isinstance(response["daily_units"], dict):
        return False

    return True


def call_api(params: dict):
    """
    Makes an API call with given parameters, handling rate limits and other errors.
    :param params: Dictionary with the parameters to be passed to the API call.
    :return: The API response if successful, None otherwise.
    """
    call_attempts = 0
    backoff_time = 60
    while call_attempts < MAX_CALL_ATTEMPTS:
        try:
            response = requests.get(API_URL, params=params)
            if response.status_code == 200:
                if validate_response(response.json()):
                    return response
            elif response.status_code == 429:
                # Use the Retry-After header to determine the cool-off time
                cool_off_time = response.headers.get("Retry-After", backoff_time)
                logging.info(
                    f"Rate limit exceeded. Waiting for {cool_off_time} seconds."
                )
                logging.info(response.text)
                time.sleep(cool_off_time)
                backoff_time *= 2
            else:
                raise Exception(
                    f"API request failed with status code {response.status_code}"
                )
        except Exception as e:
            logging.info(f"API request failed with exception {e}")
            break
        call_attempts += 1
        logging.info(f"Call attempts: {call_attempts}")
    logging.info("Max number of call attempts reached. Returning empty result")
    return response


def get_data_meteo_api(
    city: str, start_date: str, end_date: str, variable: str, model: str
) -> pd.DataFrame():
    """
    Fetches daily weather data for a specific city and variable from the Meteo API.
    :param city: The name of the city to fetch the data for. Must be a key in the \
    COORDINATES dictionary.
    :param start_date: The start date for the data in the format 'YYYY-MM-DD'.
    :param end_date: The end date for the data in the format 'YYYY-MM-DD'.
    :param variable: The weather variable to fetch the data for.
    :return: A DataFrame containing the fetched data.
    :raises ValueError: If the specified city is not in the COORDINATES dictionary.
    """
    if city not in COORDINATES.keys():
        raise ValueError("City not available")
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "models": model,
        "daily": variable,
    }
    data = call_api(params)
    if data is None:
        return None
    daily_data = data.json()["daily"]
    # Convert the daily data into a DataFrame
    df = pd.DataFrame(daily_data)
    # Add the city name as a column
    df["city"] = city
    df["time"] = pd.to_datetime(df["time"])
    return df


def calculate_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the mean and standard deviation for each prefix in the \
    DataFrame's columns.
    :param df: The DataFrame to calculate the mean and standard deviation for.
    :return: A new DataFrame with the mean and standard deviation for each prefix in the 
    original DataFrame's columns.
    """
    df = df.copy()
    # Extracting prefixes (assuming they are separated by underscores)
    prefixes = set(col.split("_")[0] for col in df.columns if "_" in col)
    for prefix in prefixes:
        relevant_cols = [col for col in df.columns if col.startswith(prefix)]
        mean_values = df[relevant_cols].mean(axis=1)
        df[prefix + "_mean"] = mean_values
        std_values = df[relevant_cols].std(axis=1)
        df[prefix + "_std"] = std_values
        df = df.drop(relevant_cols, axis=1)
    return df


def plot_data(df: pd.DataFrame, variable: str) -> None:
    """
    Plots the annual mean and dispersion of a specified variable for each city in \
    the DataFrame.
    :param df: The DataFrame containing the data to plot.
    :param variable: The variable to plot.
    :return: None
    """

    # Create a mapping from short variable names to full names
    short_to_full = {v.split("_")[0]: v for v in VARIABLES}

    # Retrieve the full name of the variable
    full_variable_name = short_to_full.get(variable, variable)

    plt.figure(figsize=(12, 8))

    # Create a distinct color for each city
    colors = plt.cm.viridis(np.linspace(0, 1, len(COORDINATES)))

    for idx, city in enumerate(COORDINATES.keys()):
        city_data = df[df["city"] == city]
        mean = city_data[f"{variable}_mean"]
        std = city_data[f"{variable}_std"]
        plt.plot(
            city_data["year"],
            mean,
            label=city,
            color=colors[idx],
            marker="o",
            linestyle="-",
            linewidth=2,
        )
        plt.fill_between(
            city_data["year"], mean - std, mean + std, color=colors[idx], alpha=0.3
        )

    plt.title(
        f'Annual Mean and Dispersion of {full_variable_name.replace("_", " ").capitalize()}',
        fontsize=14,
    )
    plt.xlabel("Year", fontsize=12)
    plt.ylabel(full_variable_name.replace("_", " ").capitalize(), fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    list_data_cities = []
    for city in COORDINATES.keys():
        logging.info(f"Getting data for {city}")
        list_data_variables = []
        for variable in VARIABLES:
            logging.info(f"Getting data for {variable} in {city}")
            for model in MODELS:
                logging.info(f"Getting data for {model} in {city}")
                data = get_data_meteo_api(
                    city, "1950-01-01", "1960-12-31", variable, model
                )
                logging.info(data.head())
                list_data_variables.append(data)
        combined_df = list_data_variables[0]
        for i, df in enumerate(list_data_variables[1:], 1):
            combined_df = pd.merge(
                combined_df, df, on=["time", "city"], suffixes=("", f"_df{i}")
            )

        data_with_calculation = calculate_mean_std(combined_df)
        list_data_cities.append(data_with_calculation)

    df_total = pd.concat(list_data_cities)

    df_total["time"] = pd.to_datetime(df_total["time"])
    df_total["year"] = df_total["time"].dt.year

    annual_data = df_total.groupby(["city", "year"]).mean().reset_index()
    annual_data.to_csv("src/module_1/annual_data.csv", index=False)

    annual_data = pd.read_csv("src/module_1/annual_data.csv")
    annual_data = annual_data.drop(["time"], axis=1)

    # Plot the data for each variable dynamically
    variables = set(
        col.split("_")[0] for col in annual_data.columns if col not in ["city", "year"]
    )
    for variable in variables:
        plot_data(annual_data, variable)


if __name__ == "__main__":
    main()
