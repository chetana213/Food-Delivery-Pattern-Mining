import pandas as pd


def load_dataset(file_path):
    """Load the food delivery dataset."""
    return pd.read_csv(file_path)


def get_dataset_overview(df):
    """Return basic dataset information."""

    overview = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "start_date": pd.to_datetime(
            df["Order_Date"],
            format="%d-%m-%Y"
        ).min(),
        "end_date": pd.to_datetime(
            df["Order_Date"],
            format="%d-%m-%Y"
        ).max()
    }

    return overview


def get_missing_values(df):
    """Return missing-value counts for each column."""
    return df.isnull().sum()


def get_data_types(df):
    """Return data types of all columns."""
    return df.dtypes


def get_dataset_preview(df, n=10):
    """Return the first n rows."""
    return df.head(n)