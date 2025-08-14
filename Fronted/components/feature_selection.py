# Fronted/components/feature_selection.py
import io
import csv
import zipfile
import math

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper  # interactive cropper

from utils.api_client import (
    generate_histogram,
    perform_kmeans,
    extract_shape_features,
    extract_haralick_texture,      # legacy train/predict
    extract_cooccurrence_texture,
    replace_image,                 # persist cropped image to backend
    extract_haralick_features,     # NEW: table-style GLCM Haralick
)


# ---------- Helpers ----------
def _get_image_id_for_index(idx: int) -> str | None:
    ids = st.session_state.get("uploaded_image_ids")
    if ids and 0 <= idx < len(ids):
        return ids[idx]
    return None


def _init_state():
    st.session_state.setdefault("crop_active", False)
    st.session_state.setdefault("crop_index", None)
    st.session_state.setdefault("crop_aspect_label", "Free")
    st.session_state.setdefault("crop_realtime", True)


def _aspect_ratio_value(label: str):
    mapping = {"Free": None, "1:1 (Square)": (1, 1), "4:3": (4, 3), "16:9": (16, 9)}
    return mapping.get(label, None)


def _to_csv_bytes(columns, rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8")


# ---------- Main Entry ----------
def render_feature_selection():
    st.header("Feature Selection")
    _init_state()

    if "images" not in st.session_state or not st.session_state["images"]:
        st.warning("Please upload images first.")
        return

    images = st.session_state["images"]

    # =========================
    # Dedicated Crop Screen (centered)
    # =========================
    if st.session_state.get("crop_active") and st.session_state.get("crop_index") is not None:
        idx = int(st.session_state["crop_index"])
        if idx < 0 or idx >= len(images):
            st.error("Invalid image index for cropping.")
            st.session_state["crop_active"] = False
            st.session_state["crop_index"] = None
            st.rerun()

        st.subheader(f"Crop Image {idx+1}")
        with st.expander("Crop Options", expanded=False):
            st.session_state["crop_aspect_label"] = st.selectbox(
                "Aspect ratio",
                options=["Free", "1:1 (Square)", "4:3", "16:9"],
                index=["Free", "1:1 (Square)", "4:3", "16:9"].index(st.session_state["crop_aspect_label"]),
                key="crop_aspect_label_select",
            )
            st.session_state["crop_realtime"] = st.checkbox(
                "Realtime update",
                value=st.session_state["crop_realtime"],
                key="crop_realtime_checkbox",
            )

        img = images[idx]
        aspect = _aspect_ratio_value(st.session_state["crop_aspect_label"])

        # Centered container for cropper
        center_col = st.columns([1, 2, 1])[1]
        with center_col:
            cropped_raw = st_cropper(
                img,
                aspect_ratio=aspect,
                realtime_update=st.session_state["crop_realtime"],
                return_type="image",
                key=f"cropper_full_{idx}",
            )

        # Normalize to PIL.Image (no external preview)
        final_crop = None
        if isinstance(cropped_raw, Image.Image):
            final_crop = cropped_raw
        elif isinstance(cropped_raw, np.ndarray):
            try:
                final_crop = Image.fromarray(cropped_raw)
            except Exception:
                final_crop = None
        elif cropped_raw is not None:
            try:
                final_crop = Image.fromarray(np.array(cropped_raw))
            except Exception:
                final_crop = None

        # Centered buttons directly under cropper
        with center_col:
            btn_cols = st.columns([1, 1])
            with btn_cols[0]:
                if st.button("✖️ Cancel", key="crop_cancel"):
                    st.session_state["crop_active"] = False
                    st.session_state["crop_index"] = None
                    st.rerun()
            with btn_cols[1]:
                if st.button("✅ Crop & Save", key="crop_confirm"):
                    if final_crop is None:
                        st.error("No cropped image yet. Please adjust the crop box, then try again.")
                    else:
                        try:
                            # 1) Replace locally so ALL downstream steps use cropped image
                            st.session_state["images"][idx] = final_crop

                            # 2) Persist to backend so processing endpoints read the cropped file
                            image_id = _get_image_id_for_index(idx)
                            if image_id:
                                replace_image(image_id, final_crop)
                                st.success(f"Image {idx+1} cropped and saved.")
                            else:
                                st.warning("Could not map image to backend image_id; saved locally only.")
                        except Exception as e:
                            st.error(f"Failed to crop image {idx+1}: {e}")
                        finally:
                            # Exit crop screen back to tabs
                            st.session_state["crop_active"] = False
                            st.session_state["crop_index"] = None
                            st.rerun()

        # Stop rendering further UI on the crop screen
        return

    # =========================
    # Normal Tabs (when not cropping)
    # =========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Histogram Analysis", "k-means Clustering", "Shape Features", "Haralick Texture", "Co-Occurrence Texture"]
    )

    # --- Histogram Analysis ---
    with tab1:
        st.subheader("Histogram Analysis")
        hist_type = st.radio("Histogram Type", options=["Black and White", "Colored"], key="hist_type")
        all_images = st.checkbox("Generate histogram for all images", key="hist_all")
        selected_index = st.selectbox(
            "Select image", options=list(range(len(images))), format_func=lambda x: f"Image {x+1}", key="hist_image_idx"
        )

        if all_images:
            st.markdown("### Selected Images")
            cols = st.columns(4)
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    st.image(img, caption=f"Image {idx+1}", width=150)
            st.info("Cropping works on a single selected image. Uncheck 'Generate histogram for all images' to crop.")
        else:
            img = images[selected_index]
            st.image(img, caption=f"Preview: Image {selected_index+1}", width=250)

            # The ONLY Crop button — takes you to the centered crop screen
            if st.button(f"Crop {selected_index+1}", key="btn_crop_under_hist"):
                st.session_state["crop_active"] = True
                st.session_state["crop_index"] = int(selected_index)
                st.rerun()

        if st.button("Generate Histogram", key="btn_histogram"):
            params = {"hist_type": hist_type, "image_index": selected_index, "all_images": all_images}
            hist_list = generate_histogram(params)
            if hist_list:
                st.session_state["histogram_images"] = hist_list
                st.success(f"Generated {len(hist_list)} histogram{'s' if len(hist_list) > 1 else ''}!")
            else:
                st.warning("No histograms were generated.")

        if st.session_state.get("histogram_images"):
            st.subheader("Generated Histograms")
            cols = st.columns(4)
            for i, img_bytes in enumerate(st.session_state["histogram_images"]):
                with cols[i % 4]:
                    st.image(img_bytes, caption=f"Histogram {i+1}", width=200)
                    if st.button(f"Enlarge {i+1}", key=f"enlarge_hist_{i}"):
                        st.session_state["fullscreen_image"] = img_bytes
                        st.session_state["fullscreen_section"] = "histogram"

            if st.button("Download All Histograms", key="download_all_histograms"):
                zip_data = create_histogram_zip(st.session_state["histogram_images"])
                st.session_state["histogram_zip"] = zip_data
                st.success("Histograms ZIP created. Click below to download.")

            if "histogram_zip" in st.session_state:
                st.download_button(
                    label="Download Histograms ZIP",
                    data=st.session_state["histogram_zip"],
                    file_name="histograms.zip",
                    mime="application/zip",
                )

    # --- k-means Clustering ---
    with tab2:
        st.subheader("k-means Clustering")
        cluster_count = st.number_input("Number of Clusters", min_value=2, max_value=10, value=2, key="kmeans_k")
        random_state = st.number_input("Random Seed", min_value=0, max_value=100, value=42, key="kmeans_rs")

        st.markdown("### Image Selection")
        all_images_checkbox = st.checkbox("Select all images", key="kmeans_all_images")

        if all_images_checkbox:
            selected_indices = list(range(len(images)))
            st.info(f"All {len(images)} images selected")
        else:
            selected_indices = st.multiselect(
                "Select images for clustering",
                options=list(range(len(images))),
                format_func=lambda x: f"Image {x+1}",
                key="kmeans_image_indices",
            )

        if st.button("Perform K-means", key="btn_kmeans"):
            if not images:
                st.error("No images have been uploaded. Please upload images first.")
            elif not selected_indices and not all_images_checkbox:
                st.error("Select at least one image or check 'Select all images'.")
            else:
                try:
                    params = {
                        "n_clusters": cluster_count,
                        "random_state": random_state,
                        "selected_images": selected_indices if not all_images_checkbox else [],
                        "use_all_images": all_images_checkbox,
                    }
                    plot_bytes, assignments = perform_kmeans(params)
                    st.image(plot_bytes, caption="K-means Clustering", width=400)
                    st.write("Cluster Assignments:")

                    if "metadata_df" in st.session_state and "image_id_col" in st.session_state:
                        metadata = st.session_state["metadata_df"]
                        id_col = st.session_state["image_id_col"]
                        image_ids = metadata[id_col].tolist()
                        for idx, label in enumerate(assignments):
                            image_name = image_ids[idx] if idx < len(image_ids) else f"Image {idx+1}"
                            st.write(f"{image_name} → Cluster {label}")
                    else:
                        for idx, label in enumerate(assignments):
                            st.write(f"Image {idx+1} → Cluster {label}")
                except Exception as e:
                    st.error(f"Error performing K-means clustering: {str(e)}")

    # --- Shape Features ---
    with tab3:
        st.subheader("Shape Feature Extraction")
        shape_methods = ["HOG", "SIFT", "FAST"]
        shape_method = st.selectbox("Method", options=shape_methods, key="shape_method")
        selected_idx = st.selectbox(
            "Select Image", options=list(range(len(images))), format_func=lambda x: f"Image {x+1}", key="shape_img_idx"
        )

        if st.button("Extract Shape Features", key="btn_shape"):
            result = extract_shape_features({"method": shape_method, "image_index": selected_idx})
            st.write(f"{shape_method} Features:")
            st.write(result.get("features"))
            viz = result.get("visualization")
            if viz:
                st.image(viz, caption=f"{shape_method} Visualization", width=400)

    # --- Haralick Texture ---
    with tab4:
        st.subheader("Haralick Texture Features")

        # --- NEW: GLCM Haralick for current uploaded images (table output) ---
        st.markdown("##### Compute GLCM-based Haralick features (current images)")
        col_a, col_b = st.columns(2)
        with col_a:
            # Select images
            use_all = st.checkbox("Use all images", value=True, key="har_use_all")
            if use_all:
                image_indices = list(range(len(images)))
            else:
                image_indices = st.multiselect(
                    "Select images",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="har_img_indices",
                )
            # Levels
            levels = st.selectbox("Quantization levels", [16, 32, 64, 128, 256], index=4, key="har_levels")
            # Distances
            distances = st.multiselect("Distances (pixels)", [1, 2, 3, 5], default=[1, 2], key="har_distances")
        with col_b:
            # Angles (radians)
            angle_map = {
                "0°": 0.0,
                "45°": math.pi / 4,
                "90°": math.pi / 2,
                "135°": 3 * math.pi / 4,
            }
            angle_labels = list(angle_map.keys())
            angles_sel = st.multiselect("Angles", angle_labels, default=angle_labels, key="har_angles_lbls")
            angles = [angle_map[a] for a in angles_sel]
            # Resize
            use_resize = st.checkbox("Resize before analysis", value=False, key="har_use_resize")
            if use_resize:
                resize_width = st.number_input("Resize width", min_value=32, max_value=2048, value=256, step=16, key="har_w")
                resize_height = st.number_input("Resize height", min_value=32, max_value=2048, value=256, step=16, key="har_h")
            else:
                resize_width = None
                resize_height = None

        props = st.multiselect(
            "Properties",
            ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"],
            default=["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"],
            key="har_props",
        )
        avg = st.checkbox("Average across all distances/angles", value=True, key="har_avg")

        if st.button("Compute Haralick (GLCM)", key="btn_haralick_table"):
            if not image_indices:
                st.error("Please select at least one image.")
            else:
                with st.spinner("Computing Haralick features..."):
                    try:
                        payload = {
                            "image_indices": image_indices,
                            "levels": int(levels),
                            "distances": distances if distances else [1],
                            "angles": angles if angles else [0.0],
                            "resize_width": int(resize_width) if resize_width else None,
                            "resize_height": int(resize_height) if resize_height else None,
                            "average_over_angles": bool(avg),
                            "properties": props,
                        }
                        result = extract_haralick_features(payload)
                        cols = result.get("columns", [])
                        rows = result.get("rows", [])
                        if cols and rows:
                            st.success(f"Computed features for {len(rows)} image(s).")
                            st.dataframe(
                                {cols[i]: [row[i] for row in rows] for i in range(len(cols))},
                                use_container_width=True,
                            )
                            csv_bytes = _to_csv_bytes(cols, rows)
                            st.download_button(
                                "Download CSV",
                                data=csv_bytes,
                                file_name="haralick_features.csv",
                                mime="text/csv",
                                key="btn_dl_haralick_csv",
                            )
                        else:
                            st.warning("No features returned.")
                    except Exception as e:
                        st.error(f"Haralick computation failed: {e}")

        st.divider()

        # --- Legacy demo: train/predict via multipart ---
        st.markdown("##### Legacy demo: Train/Predict (multipart upload)")
        train_imgs = st.file_uploader(
            "Training Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="har_train_imgs"
        )
        train_csv = st.file_uploader("Training Labels CSV", type="csv", key="har_train_csv")
        test_imgs = st.file_uploader(
            "Test Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="har_test_imgs"
        )

        if st.button("Extract Haralick (legacy)", key="btn_haralick"):
            labels, preds = extract_haralick_texture(
                {"train_images": train_imgs, "train_labels": train_csv, "test_images": test_imgs}
            )
            st.write("Predicted Labels:")
            for i, p in enumerate(preds):
                st.write(f"Test Image {i+1}: {p}")

    # --- Co-Occurrence Texture ---
    with tab5:
        st.subheader("Co-Occurrence Texture Features")
        tex_idx = st.selectbox(
            "Select Image", options=list(range(len(images))), format_func=lambda x: f"Image {x+1}", key="tex_img_idx"
        )
        if st.button("Extract Co-Occurrence", key="btn_cooccurrence"):
            features = extract_cooccurrence_texture({"image_index": tex_idx})
            st.write("Texture Features:")
            st.write(features)

    # --- Fullscreen Image Modal (histograms)
    if st.session_state.get("fullscreen_image"):
        st.image(st.session_state["fullscreen_image"], caption="Fullscreen Image", use_column_width=True)
        if st.button("Close", key="close_fullscreen"):
            st.session_state["fullscreen_image"] = None
            st.session_state["fullscreen_section"] = None

    # --- Navigation ---
    col1, col2 = st.columns([2, 1])
    with col2:
        if st.button("Next: Statistical Analysis", key="next_stats"):
            st.session_state["active_section"] = "Statistics Analysis"


# ---------- Utility ----------
def create_histogram_zip(hist_list):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_bytes in enumerate(hist_list):
            zip_file.writestr(f"histogram_{i+1}.png", img_bytes)
    return zip_buffer.getvalue()


if __name__ == "__main__":
    render_feature_selection()
