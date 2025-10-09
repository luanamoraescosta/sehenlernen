# Fronted/components/data_input.py

import io
from PIL import Image
import streamlit as st
from utils import api_client


def render_data_input():
    st.header("Data Input")

    # Ensure keys exist
    st.session_state.setdefault("images", [])                # PIL previews for UI + next screen
    st.session_state.setdefault("uploaded_image_ids", [])    # IDs stored on backend

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

    zip_file = st.file_uploader(
        "Choose a ZIP file containing images",
        type=["zip"],
        key="upload_zip_file"
    )

    # Live local previews (persist to session so Feature Selection can see them)
    if image_files:
        try:
            st.session_state["images"] = [
                Image.open(io.BytesIO(f.getbuffer())) for f in image_files
            ]
        except Exception as e:
            st.error(f"Failed to read selected images for preview: {e}")
            st.session_state["images"] = []

    # Manual upload button (kept as-is)
    if image_files or zip_file:
        if st.button("Upload Images", key="btn_upload_images"):
            with st.spinner("Uploading images..."):
                try:
                    image_ids = api_client.upload_images(image_files, zip_file)
                    st.session_state["uploaded_image_ids"] = image_ids
                    st.success(f"Uploaded {len(image_ids)} images successfully.")
                except Exception as e:
                    st.error(f"Failed to upload images: {e}")

    # Show previews if we have any
    if st.session_state["images"]:
        st.write("Preview of selected images:")
        cols = st.columns(6)
        for i, img in enumerate(st.session_state["images"]):
            with cols[i % 6]:
                st.image(img, caption=f"Image {i+1}", width=200)

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
                        # Make the ZIP downloadable
                        st.session_state["extractor_zip"] = zip_bytes
                        # IDs available for downstream steps
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
        # If user selected files but didn't press "Upload Images", upload now so backend has them
        if image_files and not st.session_state.get("uploaded_image_ids"):
            with st.spinner("Uploading images..."):
                try:
                    image_ids = api_client.upload_images(image_files)
                    st.session_state["uploaded_image_ids"] = image_ids
                    st.success(f"Uploaded {len(image_ids)} images successfully.")
                except Exception as e:
                    st.error(f"Failed to upload images: {e}")
                    return  # stay on this page

        # Require either previews or uploaded IDs to proceed
        if st.session_state.get("images") or st.session_state.get("uploaded_image_ids"):
            st.session_state["active_section"] = "Feature Selection"
        else:
            st.warning("Please upload at least one image before continuing.")
