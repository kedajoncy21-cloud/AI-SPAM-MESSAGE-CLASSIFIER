"""
Module to load and inspect the spam dataset.
"""

from typing import Optional
import pandas as pd


def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV file and return a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame or None if error occurs
    """
    try:
        data = pd.read_csv(filepath)
        print("Data loaded successfully")
        print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        return data
    except FileNotFoundError:
        print("Error: File not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
        return None
    except Exception as error:  # pylint: disable=broad-except
        print(f"Unexpected error: {error}")
        return None


def main() -> None:
    """
    Main function to execute data loading.
    """
    dataframe = load_data("../data/spam.csv")

    if dataframe is not None:
        print(dataframe.head())


if __name__ == "__main__":
    main()