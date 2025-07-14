# frontend/components/data_input.py

import streamlit as st
from PIL import Image
import io

from utils.api_client import (
    upload_images,
    upload_metadata_file,
    configure_metadata,
    filter_sampling,
    stratified_sampling,
)

def render_data_input():
    st.header("Data Input")

    # --- File Upload ---
    uploaded_images = st.file_uploader(
        "Select image files",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"],
        key="upload_images"
    )
    # Send to backend once
    if uploaded_images and "uploaded_image_ids" not in st.session_state:
        st.session_state["uploaded_image_ids"] = upload_images(uploaded_images)

    # Store PIL images locally so next screen can access them
    if uploaded_images:
        st.session_state["images"] = [
            Image.open(io.BytesIO(f.getbuffer())) for f in uploaded_images
        ]

    # --- Metadata Configuration ---
    uploaded_csv = st.file_uploader(
        "Upload Metadata File (optional)",
        type=["csv", "xlsx"],
        key="upload_metadata"
    )
    if uploaded_csv:
        st.markdown("### Metadata Configuration")
        delimiter = st.selectbox("Select delimiter", [",", ";", "\t"], key="delimiter")
        decimal_sep = st.selectbox("Select decimal separator", [".", ","], key="decimal_sep")

        if st.button("Load Metadata", key="load_metadata"):
            st.session_state["metadata_columns"] = upload_metadata_file(
                uploaded_csv, delimiter, decimal_sep
            )

        if "metadata_columns" in st.session_state:
            cols = st.session_state["metadata_columns"]
            image_id_col = st.selectbox(
                "Select column for Image ID",
                options=cols,
                key="image_id_col"
            )

            col_mapping = {}
            for col in cols:
                if col != image_id_col:
                    dtype = st.selectbox(
                        f"Data type for {col}",
                        ["Numeric", "Categorical", "Datetime", "String"],
                        key=f"dtype_{col}"
                    )
                    col_mapping[col] = dtype

            if st.button("Configure Metadata", key="config_metadata"):
                configure_metadata(image_id_col, col_mapping)
                st.session_state["metadata_configured"] = True
                st.success("Column mapping completed!")

    # --- Sampling Methods ---
    sampling_method = st.selectbox(
        "Choose a sampling method",
        ["Use All Images", "Filter by Metadata", "Stratified Sampling by Metadata"],
        key="sampling_method"
    )

    if sampling_method == "Use All Images" and uploaded_images:
        st.session_state["sample_size"] = len(uploaded_images)
        st.write(f"All {len(uploaded_images)} images will be used.")

    elif sampling_method == "Filter by Metadata":
        if st.session_state.get("metadata_configured"):
            filter_cols = st.multiselect(
                "Select columns to filter",
                options=st.session_state["metadata_columns"],
                key="filter_cols"
            )
            filter_values = {}
            for col in filter_cols:
                options = st.session_state.get("metadata_unique_values", {}).get(col, [])
                values = st.multiselect(
                    f"Select values for {col}",
                    options=options,
                    key=f"filter_vals_{col}"
                )
                filter_values[col] = values

            if st.button("Apply Filter", key="apply_filter"):
                sampled_ids = filter_sampling(filter_values)
                st.session_state["sample_size"] = len(sampled_ids)
                st.write(f"After filtering: {len(sampled_ids)} images")
        else:
            st.warning("Please configure metadata first.")

    elif sampling_method == "Stratified Sampling by Metadata":
        if st.session_state.get("metadata_configured"):
            categorical_cols = [
                col for col, dt in st.session_state.get("col_mapping", {}).items()
                if dt == "Categorical"
            ]
            if categorical_cols:
                target_col = st.selectbox(
                    "Select target column for stratification",
                    options=categorical_cols,
                    key="strat_target"
                )
                sample_size = st.number_input(
                    "Desired sample size",
                    min_value=1,
                    max_value=len(uploaded_images) if uploaded_images else 1,
                    value=len(uploaded_images) if uploaded_images else 1,
                    key="strat_size"
                )
                if st.button("Apply Stratified Sampling", key="apply_stratified"):
                    sampled_ids = stratified_sampling(target_col, sample_size)
                    st.session_state["sample_size"] = len(sampled_ids)
                    st.write(f"Sampled {len(sampled_ids)} images")
            else:
                st.warning("No categorical columns found for stratified sampling.")
        else:
            st.warning("Please configure metadata first.")

    # --- Preview Uploaded Images ---
    if "images" in st.session_state:
        st.subheader("Preview of Uploaded Images")
        cols = st.columns(6)
        for i, img in enumerate(st.session_state["images"]):
            with cols[i % 6]:
                st.image(img, caption=f"Image {i+1}", width=200)

    # --- Navigation ---
    if st.button("Next: Feature Selection", key="next_to_feature"):
        st.session_state["active_section"] = "Feature Selection"
