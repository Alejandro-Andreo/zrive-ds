import unittest
from unittest.mock import patch, Mock
import pandas as pd
import requests
from src.module_1.module_1_meteo_api import (
    validate_response,
    calculate_mean_std,
    call_api,
)


class TestValidateResponse(unittest.TestCase):
    def test_validate_response_valid(self):
        response = {
            "latitude": 40.416775,
            "longitude": -3.703790,
            "generationtime_ms": 2.2119,
            "timezone": "Europe/Madrid",
            "timezone_abbreviation": "CEST",
            "daily": {},
            "daily_units": {},
        }
        self.assertTrue(validate_response(response))

    def test_validate_response_invalid(self):
        response = {"invalid_key": "invalid_value"}
        self.assertFalse(validate_response(response))


class TestCalculateMeanStd(unittest.TestCase):
    def test_calculate_mean_std(self):
        data = {
            "time": ["2022-01-01", "2022-01-02"],
            "temperature_2m_AAAAAAAA": [10, 20],
            "temperature_2m_BBBBBBBB": [15, 25],
            "city": ["City1", "City1"],
        }
        df = pd.DataFrame(data)
        result = calculate_mean_std(df)
        self.assertIn("temperature_mean", result.columns)
        self.assertIn("temperature_std", result.columns)
        self.assertEqual(result["temperature_mean"][0], 12.5)
        self.assertEqual(result["temperature_mean"][1], 22.5)


class TestCallAPI(unittest.TestCase):
    @patch("src.module_1.module_1_meteo_api.requests.get")
    def test_call_api_successful(self, mock_get):
        # Mock a successful API response
        mock_response = Mock()
        mock_get.return_value = mock_response
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "latitude": 40.416775,
            "longitude": -3.703790,
            "generationtime_ms": 2.2119,
            "timezone": "Europe/Madrid",
            "timezone_abbreviation": "CEST",
            "daily": {},
            "daily_units": {},
        }
        params = {
            # Mocked parameters
        }
        result = call_api(params)
        self.assertIsNotNone(result)
        self.assertEqual(result.status_code, 200)

    @patch("src.module_1.module_1_meteo_api.requests.get")
    def test_call_api_failure(self, mock_get):
        # Mock a failed API response
        mock_response = Mock()
        mock_get.return_value = mock_response
        mock_response.status_code = 404

        result = call_api({})

        self.assertIsNone(result)

    @patch("src.module_1.module_1_meteo_api.requests.get")
    def test_call_api_rate_limit_exceeded_then_success(self, mock_get):
        # Mock the first response as a rate limit error, then a successful response
        mock_response_rate_limit = Mock()
        mock_response_rate_limit.raise_for_status.side_effect = (
            requests.exceptions.HTTPError()
        )
        mock_response_rate_limit.status_code = 429
        mock_response_rate_limit.headers = {
            "Retry-After": 5
        }  # Simulate a short retry after period for the test

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "latitude": 40.416775,
            "longitude": -3.703790,
            "generationtime_ms": 2.2119,
            "timezone": "Europe/Madrid",
            "timezone_abbreviation": "CEST",
            "daily": {},
            "daily_units": {},
        }

        mock_get.side_effect = [mock_response_rate_limit, mock_response_success]

        params = {}  # Mocked parameters
        result = call_api(params)

        self.assertIsNotNone(result)
        self.assertEqual(result.status_code, 200)

    @patch("src.module_1.module_1_meteo_api.logging.error")
    @patch("src.module_1.module_1_meteo_api.requests.get")
    def test_call_api_connection_error(self, mock_get, mock_log_error):
        # Mock a connection error
        mock_get.side_effect = requests.exceptions.ConnectionError()

        params = {}  # Mocked parameters
        result = call_api(params)

        self.assertIsNone(result)
        # Assert that logging.error was called at least once
        self.assertTrue(mock_log_error.called, "Expected logging.error to be called")
        # Check if any of the calls to logging.error start with "Connection error:"
        error_message_starts_correctly = any(
            call_arg[0][0].startswith("Connection error:")
            for call_arg in mock_log_error.call_args_list
        )
        self.assertTrue(
            error_message_starts_correctly,
            "No log message starts with 'Connection error:'",
        )

    @patch("src.module_1.module_1_meteo_api.logging.error")
    @patch("src.module_1.module_1_meteo_api.requests.get")
    def test_call_api_timeout(self, mock_get, mock_log_error):
        # Mock a timeout error
        mock_get.side_effect = requests.exceptions.Timeout()

        params = {}  # Mocked parameters
        result = call_api(params)

        self.assertIsNone(result)

        self.assertTrue(mock_log_error.called, "Expected logging.error to be called")
        error_message_starts_correctly = any(
            call_arg[0][0].startswith("Timeout error:")
            for call_arg in mock_log_error.call_args_list
        )
        self.assertTrue(
            error_message_starts_correctly,
            "No log message starts with 'Timeout error:'",
        )


if __name__ == "__main__":
    unittest.main()
