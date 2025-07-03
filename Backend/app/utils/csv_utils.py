import io
import pandas as pd
from fastapi import UploadFile

# Utility functions for CSV/Excel handling

def read_metadata_file(file: UploadFile, delimiter: str = ',', decimal: str = '.') -> pd.DataFrame:
    """
    Read an uploaded CSV or Excel file into a pandas DataFrame.
    """
    contents = file.file.read()
    name = file.filename.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents), delimiter=delimiter, decimal=decimal)
    else:
        df = pd.read_excel(io.BytesIO(contents))
    return df


def df_to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    """
    Convert a DataFrame to CSV bytes.
    """
    buf = io.BytesIO()
    df.to_csv(buf, index=index)
    return buf.getvalue()
