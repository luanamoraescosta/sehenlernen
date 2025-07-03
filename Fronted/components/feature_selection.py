import streamlit as st
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

from utils.api_client import (
    generate_histogram,
    perform_kmeans,
    extract_shape_features,
    extract_haralick_texture,
    extract_cooccurrence_texture,
)

# Feature Selection page component
def render_feature_selection():
    st.header("Feature Selection")
    
    if "images" not in st.session_state or not st.session_state["images"]:
        st.warning("Please upload images first.")
        return
    
    images = st.session_state["images"]
    sampled_ids = st.session_state.get("sampled_image_ids", None)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Histogram Analysis",
        "k-means Clustering",
        "Shape Features",
        "Haralick Texture",
        "Co-Occurrence Texture"
    ])

    # --- Histogram Analysis ---
    with tab1:
        st.subheader("Histogram Analysis")
        hist_type = st.radio(
            "Histogram Type",
            options=["Black and White", "Colored"],
            key="hist_type"
        )
        all_images = st.checkbox("Generate histogram for all images", key="hist_all")
        selected_index = st.selectbox(
            "Select image",
            options=list(range(len(images))),
            format_func=lambda x: f"Image {x+1}",
            key="hist_image_idx"
        )

        # Preview
        img = images[selected_index]
        st.image(img, caption=f"Preview: Image {selected_index+1}", width=200)

        if st.button("Generate Histogram", key="btn_histogram"):
            params = {
                "hist_type": hist_type,
                "image_index": selected_index,
                "all_images": all_images
            }
            hist_list = generate_histogram(params)
            st.session_state["histogram_images"] = hist_list
            st.success(f"Generated {len(hist_list)} histograms!")

        if st.session_state.get("histogram_images"):
            st.subheader("Generated Histograms")
            cols = st.columns(4)
            for i, img_bytes in enumerate(st.session_state["histogram_images"]):
                with cols[i % 4]:
                    st.image(img_bytes, caption=f"Histogram {i+1}", width=200)
                    if st.button(f"Enlarge {i+1}", key=f"enlarge_hist_{i}"):
                        st.session_state["fullscreen_image"] = img_bytes
                        st.session_state["fullscreen_section"] = "histogram"

    # --- k-means Clustering ---
    with tab2:
        st.subheader("k-means Clustering")
        cluster_count = st.number_input(
            "Number of Clusters", min_value=2, max_value=10, value=2, key="kmeans_k"
        )
        random_state = st.number_input(
            "Random State", min_value=0, max_value=100, value=42, key="kmeans_rs"
        )

        if st.button("Perform k-means", key="btn_kmeans"):
            params = {"n_clusters": cluster_count, "random_state": random_state}
            plot_bytes, assignments = perform_kmeans(params)
            st.image(plot_bytes, caption="k-means Clustering", width=400)
            st.write("Cluster Assignments:")
            for idx, label in enumerate(assignments):
                st.write(f"Image {idx+1} → Cluster {label}")

    # --- Shape Features ---
    with tab3:
        st.subheader("Shape Feature Extraction")
        shape_methods = ["HOG", "SIFT", "FAST"]
        shape_method = st.selectbox("Method", options=shape_methods, key="shape_method")
        selected_idx = st.selectbox(
            "Select Image",
            options=list(range(len(images))),
            format_func=lambda x: f"Image {x+1}",
            key="shape_img_idx"
        )

        if st.button("Extract Shape Features", key="btn_shape"): 
            result = extract_shape_features({
                "method": shape_method,
                "image_index": selected_idx
            })
            # result contains 'features' and optionally 'visualization'
            st.write(f"{shape_method} Features:")
            st.write(result.get("features"))
            viz = result.get("visualization")
            if viz:
                st.image(viz, caption=f"{shape_method} Visualization", width=400)

    # --- Haralick Texture ---
    with tab4:
        st.subheader("Haralick Texture Features")
        train_imgs = st.file_uploader(
            "Training Images", type=["png","jpg","jpeg"], accept_multiple_files=True, key="har_train_imgs"
        )
        train_csv = st.file_uploader(
            "Training Labels CSV", type=["csv"], key="har_train_csv"
        )
        test_imgs = st.file_uploader(
            "Test Images", type=["png","jpg","jpeg"], accept_multiple_files=True, key="har_test_imgs"
        )

        if st.button("Extract Haralick", key="btn_haralick"):
            labels, preds = extract_haralick_texture({
                "train_images": train_imgs,
                "train_labels": train_csv,
                "test_images": test_imgs
            })
            st.write("Predicted Labels:")
            for i, p in enumerate(preds):
                st.write(f"Test Image {i+1}: {p}")

    # --- Co-Occurrence Texture ---
    with tab5:
        st.subheader("Co-Occurrence Texture Features")
        tex_idx = st.selectbox(
            "Select Image",
            options=list(range(len(images))),
            format_func=lambda x: f"Image {x+1}",
            key="tex_img_idx"
        )
        if st.button("Extract Co-Occurrence", key="btn_cooccurrence"):
            features = extract_cooccurrence_texture({"image_index": tex_idx})
            st.write("Texture Features:")
            st.write(features)

    # --- Navigation ---
    col1, col2 = st.columns([2,1])
    with col2:
        if st.button("Next: Statistical Analysis", key="next_stats"):
            st.session_state["active_section"] = "Statistics Analysis"
