import streamlit as st
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import zipfile

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

         # Preview section
        if all_images:
            st.markdown("### Selected Images")
            cols = st.columns(4)
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    st.image(img, caption=f"Image {idx+1}", width=150)
        else:
            img = images[selected_index]
            st.image(img, caption=f"Preview: Image {selected_index+1}", width=200)

        if st.button("Generate Histogram", key="btn_histogram"):
            params = {
                "hist_type": hist_type,
                "image_index": selected_index,
                "all_images": all_images
            }
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

            # Botão para gerar o ZIP
            if st.button("Download All Histograms", key="download_all_histograms"):
                zip_data = create_histogram_zip(st.session_state["histogram_images"])
                st.session_state["histogram_zip"] = zip_data
                st.success("Histograms ZIP created. Click below to download.")

            # Botão de download (aparece apenas se o ZIP foi gerado)
            if "histogram_zip" in st.session_state:
                st.download_button(
                    label="Download Histograms ZIP",
                    data=st.session_state["histogram_zip"],
                    file_name="histograms.zip",
                    mime="application/zip"
                )


    # --- k-means Clustering ---
    with tab2:
        st.subheader("k-means Clustering")
        cluster_count = st.number_input(
            "Number of Clusters", min_value=2, max_value=10, value=2, key="kmeans_k"
        )
        random_state = st.number_input(
            "Random Seed", min_value=0, max_value=100, value=42, key="kmeans_rs"
        )
        
        # Image selection section
        st.markdown("### Image Selection")
        all_images_checkbox = st.checkbox("Select all images", key="kmeans_all_images")
        
        # Ensure selected_indices is always a list
        if all_images_checkbox:
            selected_indices = list(range(len(images)))
            st.info(f"All {len(images)} images selected")
        else:
            selected_indices = st.multiselect(
                "Select images for clustering",
                options=list(range(len(images))),
                format_func=lambda x: f"Image {x+1}",
                key="kmeans_image_indices"
            )
        
        # Explicit validation before creating params
        if st.button("Perform K-means", key="btn_kmeans"):
            # Check minimum requirements
            if not images:
                st.error("No images have been uploaded. Please upload images first.")
            elif not selected_indices and not all_images_checkbox:
                st.error("Select at least one image or check 'Select all images'.")
            else:
                try:
                    # Safe params initialization
                    params = {
                        "n_clusters": cluster_count,
                        "random_state": random_state,
                        "selected_images": selected_indices if not all_images_checkbox else [],
                        "use_all_images": all_images_checkbox
                    }
                    
                    plot_bytes, assignments = perform_kmeans(params)
                    st.image(plot_bytes, caption="K-means Clustering", width=400)
                    st.write("Cluster Assignments:")
                    
                    # Show results with image names if metadata is available
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


    # --- Fullscreen Image Modal ---
    if st.session_state.get("fullscreen_image"):
        st.image(st.session_state["fullscreen_image"], caption="Fullscreen Image", use_container_width=True)
        if st.button("Close", key="close_fullscreen"):
            st.session_state["fullscreen_image"] = None
            st.session_state["fullscreen_section"] = None


    # --- Navigation ---
    col1, col2 = st.columns([2,1])
    with col2:
        if st.button("Next: Statistical Analysis", key="next_stats"):
            st.session_state["active_section"] = "Statistics Analysis"

def create_histogram_zip(hist_list):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_bytes in enumerate(hist_list):
            zip_file.writestr(f"histogram_{i+1}.png", img_bytes)
    return zip_buffer.getvalue()

if __name__ == "__main__":
    render_feature_selection()