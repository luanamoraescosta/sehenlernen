# Fronted/components/data_input.py

import streamlit as st
from utils import api_client


def render_data_input():
    st.header("Data Input")

    # -------------------------
    # Section 1: Upload Images
    # -------------------------
    st.subheader("Upload Images")
    image_files = st.file_uploader(
        "Choose image files",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="upload_images_files"
    )

    if image_files:
        if st.button("Upload Images", key="btn_upload_images"):
            with st.spinner("Uploading images..."):
                try:
                    image_ids = api_client.upload_images(image_files)
                    st.session_state["uploaded_image_ids"] = image_ids
                    st.success(f"Uploaded {len(image_ids)} images successfully.")
                except Exception as e:
                    st.error(f"Failed to upload images: {e}")

    st.markdown("---")

    # ------------------------------------------------------
    # Section 2: EXTRACTOR (CSV of Image URLs) — NO METADATA
    # ------------------------------------------------------
    st.subheader("Extractor: Download Images from CSV of URLs")
    extractor_csv = st.file_uploader(
        "Upload CSV containing image URLs (one URL per row). We auto-detect the URL column (e.g., url, image_url, link) or use the first column.",
        type=["csv"],
        key="extractor_csv_file"
    )

    if extractor_csv:
        if st.button("Extract Images from CSV", key="btn_extract_csv"):
            with st.spinner("Extracting images from URLs..."):
                try:
                    zip_bytes, image_ids, errors = api_client.extract_images_from_csv(extractor_csv)
                    if zip_bytes:
                        st.session_state["extractor_zip"] = zip_bytes
                        st.session_state["extractor_image_ids"] = image_ids
                        # Also expose to the rest of the app (Feature Selection etc.)
                        st.session_state["uploaded_image_ids"] = image_ids
                        st.success(f"Extracted {len(image_ids)} images successfully.")
                        if errors:
                            st.warning(f"{len(errors)} errors occurred during extraction.")
                            for err in errors:
                                st.text(f"- {err}")
                    else:
                        st.error("No ZIP file returned from server.")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

    if "extractor_zip" in st.session_state:
        st.download_button(
            label="Download Extracted Images ZIP",
            data=st.session_state["extractor_zip"],
            file_name="extracted_images.zip",
            mime="application/zip",
            key="btn_download_extractor_zip"
        )

    st.markdown("---")

    # Navigation
    if st.button("Next: Feature Selection", key="btn_next_feature_selection"):
        st.session_state["active_section"] = "Feature Selection"
