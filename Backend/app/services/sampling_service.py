import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from app.services.data_service import metadata_df, image_id_col


def filter_images_by_metadata(filters: dict) -> list[str]:
    """
    Filter metadata_df by filters dict and return matching image IDs.
    filters: { column_name: [values] }
    """
    if metadata_df is None or image_id_col is None:
        raise ValueError("Metadata not configured")

    df = metadata_df.copy()
    for col, values in filters.items():
        if values:
            df = df[df[col].isin(values)]
    # Return image IDs from the designated column
    return df[image_id_col].astype(str).tolist()


def stratified_sample_images(target_col: str, sample_size: int) -> list[str]:
    """
    Perform stratified sampling on metadata_df based on target_col.
    Returns list of sampled image IDs.
    """
    if metadata_df is None or image_id_col is None:
        raise ValueError("Metadata not configured")

    # Ensure target_col is present
    if target_col not in metadata_df.columns:
        raise ValueError(f"Column {target_col} not in metadata")

    df = metadata_df.copy()
    y = df[target_col]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
    # Get first (and only) split
    train_idx, _ = next(sss.split(df, y))
    sampled_df = df.iloc[train_idx]
    return sampled_df[image_id_col].astype(str).tolist()
