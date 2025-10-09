# Fronted/components/feature_selection.py
import io
import csv
import zipfile
import math
import base64
import logging
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper  # interactive cropper
import time

from utils.api_client import (
    generate_histogram,
    perform_kmeans,
    extract_shape_features,
    extract_haralick_texture,      # train/predict workflow (kept, just renamed in UI)
    extract_cooccurrence_texture,
    replace_image,                 # persist cropped image to backend
    extract_haralick_features,     # table-style GLCM Haralick
    extract_lbp_features,          # NEW: LBP feature extraction
    extract_contours,              # NEW: Contour extraction
    extract_hog_features,          # NEW: HOG convenience call
    extract_sift_features,
    extract_edge_features,
    similarity_search,             # NEW: Similarity search
    get_similarity_methods,        # NEW: Get available methods
    precompute_similarity_features, # NEW: Precompute features
    extract_image_embedding,       # NEW: Image embedding extraction
)

# ---------- Helpers ----------
def _get_image_id_for_index(idx: int) -> str | None:
    ids = st.session_state.get("uploaded_image_ids")
    if ids and 0 <= idx < len(ids):
        return ids[idx]
    return None


def _ensure_backend_sync():
    """Ensure backend is synchronized with frontend images by re-uploading if needed."""
    try:
        from utils.api_client import get_current_image_ids, clear_all_backend_images, upload_images
        
        # Get current state
        frontend_images = st.session_state.get("images", [])
        backend_ids = get_current_image_ids()
        
        # If counts don't match, clean backend and re-upload
        if len(frontend_images) != len(backend_ids):
            # Clear backend storage
            clear_all_backend_images()
            
            # Re-upload all frontend images
            if frontend_images:
                # Convert PIL images to uploadable format
                class FileWrapper:
                    def __init__(self, name, data):
                        self.name = name
                        self.data = data
                        self.type = "image/png"
                    def getvalue(self):
                        return self.data
                
                image_files = []
                for i, img in enumerate(frontend_images):
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    file_wrapper = FileWrapper(f"image_{i+1}.png", img_bytes.getvalue())
                    image_files.append(file_wrapper)
                
                # Upload images and get new IDs
                new_ids = upload_images(image_files)
                st.session_state["uploaded_image_ids"] = new_ids
                return True
        else:
            # Counts match, just update session state
            st.session_state["uploaded_image_ids"] = backend_ids
            return True
            
    except Exception as e:
        logging.error(f"Failed to sync backend: {e}")
        return False


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

def base64_to_bytes(b64_str: str) -> bytes:
    """Convert a base64‑encoded PNG string to raw bytes that Streamlit can display."""
    return base64.b64decode(b64_str)

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
    # ADD: New tabs including Similarity Search
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "Histogram Analysis",
            "k-means Clustering", 
            "Shape Features",
            "Similarity Search",             # NEW - Added here
            "Haralick Texture",
            "Co-Occurrence Texture",
            "Local Binary Patterns (LBP)",
            "Contour Extraction",
            "SIFT & Edge Detection",
        ]
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
                    # Enlarge button (existing)
                    if st.button(f"Enlarge {i+1}", key=f"enlarge_hist_{i}"):
                        st.session_state["fullscreen_image"] = img_bytes
                        st.session_state["fullscreen_section"] = "histogram"
                    # NEW: Contours button next to Enlarge
                    if st.button(f"Contours {i+1}", key=f"contours_hist_{i}"):
                        with st.spinner("Extracting contours..."):
                            try:
                                # Map histogram index back to image index
                                # If 'all images' was used, i maps directly.
                                # Otherwise, use the selected index.
                                img_idx = i if st.session_state.get("hist_all") else st.session_state.get("hist_image_idx", 0)
                                params = {
                                    "image_index": int(img_idx),
                                    "mode": "RETR_EXTERNAL",
                                    "method": "CHAIN_APPROX_SIMPLE",
                                    "min_area": 10,
                                    "return_bounding_boxes": True,
                                    "return_hierarchy": False,
                                }
                                result = extract_contours(params)
                                if result.get("visualization_bytes"):
                                    st.image(result["visualization_bytes"], caption="Contours Overlay", width=200)
                                # Show quick stats
                                areas = result.get("areas", [])
                                bbs = result.get("bounding_boxes", [])
                                if areas:
                                    st.write(f"Contours: {len(areas)}")
                                if bbs:
                                    st.write("Sample bounding box:", bbs[0])
                            except Exception as e:
                                st.error(f"Contour extraction failed: {e}")

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

        # -------------------------------
        # NEW: HOG within Histogram Tab
        # -------------------------------
        st.divider()
        st.markdown("### HOG (Histogram of Oriented Gradients)")
        st.caption(
            "HOG summarizes local gradient directions in small cells to capture object structure and edges. "
            "It’s commonly used for detection and classification. Configure the parameters and compute HOG "
            "for the selected image above."
        )
        col_hl, col_hr = st.columns(2)
        with col_hl:
            hog_orient = st.number_input("Orientation bins", min_value=3, max_value=18, value=9, step=1, key="hog_orient")
            ppc_x = st.number_input("Pixels per cell — X", min_value=4, max_value=64, value=8, step=1, key="hog_ppc_x")
            ppc_y = st.number_input("Pixels per cell — Y", min_value=4, max_value=64, value=8, step=1, key="hog_ppc_y")
        with col_hr:
            cpb_x = st.number_input("Cells per block — X", min_value=1, max_value=8, value=2, step=1, key="hog_cpb_x")
            cpb_y = st.number_input("Cells per block — Y", min_value=1, max_value=8, value=2, step=1, key="hog_cpb_y")
            use_resize = st.checkbox("Resize before HOG", value=False, key="hog_use_resize")
            if use_resize:
                hog_w = st.number_input("Resize width", min_value=32, max_value=2048, value=128, step=16, key="hog_w")
                hog_h = st.number_input("Resize height", min_value=32, max_value=2048, value=64, step=16, key="hog_h")
            else:
                hog_w = None
                hog_h = None

        if st.button("Compute HOG for Selected Image", key="btn_hog_compute"):
            with st.spinner("Computing HOG..."):
                try:
                    # NOTE: The backend currently uses default HOG params.
                    # We pass the UI params for forward-compatibility. If the backend
                    # ignores them, results reflect its defaults until we wire it up.
                    result = extract_hog_features(
                        image_index=int(selected_index),
                        orientations=int(hog_orient),
                        pixels_per_cell=(int(ppc_x), int(ppc_y)),
                        cells_per_block=(int(cpb_x), int(cpb_y)),
                        resize_width=int(hog_w) if hog_w else None,
                        resize_height=int(hog_h) if hog_h else None,
                        visualize=True,
                    )
                    feats = result.get("features", []) or []
                    st.success(f"HOG feature vector length: {len(feats)}")
                    if result.get("visualization") is not None:
                        st.image(result["visualization"], caption="HOG Visualization", width=400)

                    # Download CSV for the single vector
                    cols = [f"f{i}" for i in range(len(feats))]
                    csv_bytes = _to_csv_bytes(["image_index"] + cols, [[int(selected_index)] + feats])
                    st.download_button(
                        "Download HOG Features (CSV)",
                        data=csv_bytes,
                        file_name=f"hog_image_{int(selected_index)+1}.csv",
                        mime="text/csv",
                        key="btn_dl_hog_csv",
                    )
                except Exception as e:
                    st.error(f"HOG computation failed: {e}")

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
        
        # Create sub-tabs for different feature types
        shape_subtab1, shape_subtab2 = st.tabs(["Traditional Features", "Deep Learning Embeddings"])
        
        # Traditional shape features (HOG, SIFT, FAST)
        with shape_subtab1:
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
        
        # Deep learning embeddings
        with shape_subtab2:
            st.markdown("### Image Embedding Extraction")
            st.markdown("Extract fixed-length feature vectors using deep neural networks (ResNet, MobileNet).")
            
            # Model selection
            model_options = {
                "ResNet-50 (2048D)": "resnet50",
                "ResNet-18 (512D)": "resnet18", 
                "MobileNet-v2 (1280D)": "mobilenet_v2"
            }
            selected_model_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                key="embedding_model",
                help="Choose a pretrained model. Higher dimensions capture more information but require more storage."
            )
            selected_model = model_options[selected_model_name]
            
            # Image selection
            embed_mode = st.radio(
                "Process",
                options=["Single Image", "All Images"],
                key="embed_mode",
                horizontal=True
            )
            
            if embed_mode == "Single Image":
                embed_idx = st.selectbox(
                    "Select Image",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="embed_img_idx"
                )
                use_all = False
                image_indices = [embed_idx]
            else:
                use_all = True
                image_indices = None
                st.info(f"Will process all {len(images)} images")
            
            # Extract button
            if st.button("🎯 Extract Embeddings", key="btn_embedding", type="primary"):
                with st.spinner(f"Extracting embeddings using {selected_model_name}..."):
                    try:
                        params = {
                            "image_indices": image_indices,
                            "use_all_images": use_all,
                            "model_name": selected_model
                        }
                        result = extract_image_embedding(params)
                        
                        # Display results
                        st.success(f"✅ Extracted embeddings for {result['num_images']} image(s)")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", result['model_name'])
                        with col2:
                            st.metric("Embedding Dimension", result['embedding_dim'])
                        with col3:
                            st.metric("Images Processed", result['num_images'])
                        
                        # Show embeddings
                        st.markdown("#### Embedding Vectors")
                        
                        embeddings = result['embeddings']
                        image_ids = result['image_ids']
                        
                        # Display in expandable sections
                        for i, (emb, img_id) in enumerate(zip(embeddings, image_ids)):
                            with st.expander(f"Image {i+1} ({img_id}) - {len(emb)}D vector"):
                                # Show first 10 values as preview
                                st.write(f"**Preview (first 10 values):** {emb[:10]}")
                                st.write(f"**Shape:** {len(emb)} dimensions")
                                
                                # Show statistics
                                emb_array = np.array(emb)
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Mean", f"{emb_array.mean():.4f}")
                                with col_b:
                                    st.metric("Std Dev", f"{emb_array.std():.4f}")
                                with col_c:
                                    st.metric("L2 Norm", f"{np.linalg.norm(emb_array):.4f}")
                                
                                # Full vector as CSV download
                                csv_data = ",".join(map(str, emb))
                                st.download_button(
                                    label="📥 Download Full Vector (CSV)",
                                    data=csv_data,
                                    file_name=f"embedding_{img_id}.csv",
                                    mime="text/csv",
                                    key=f"download_emb_{i}"
                                )
                        
                        # Download all embeddings as CSV
                        if len(embeddings) > 1:
                            st.markdown("---")
                            # Create CSV with all embeddings
                            csv_rows = []
                            csv_rows.append(["image_id"] + [f"dim_{j}" for j in range(len(embeddings[0]))])
                            for img_id, emb in zip(image_ids, embeddings):
                                csv_rows.append([img_id] + emb)
                            
                            csv_buffer = io.StringIO()
                            import csv
                            writer = csv.writer(csv_buffer)
                            writer.writerows(csv_rows)
                            
                            st.download_button(
                                label="📥 Download All Embeddings (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name=f"all_embeddings_{selected_model}.csv",
                                mime="text/csv",
                                key="download_all_emb"
                            )
                            
                    except Exception as e:
                        st.error(f"Failed to extract embeddings: {str(e)}")
                        if "PyTorch is not installed" in str(e):
                            st.info("💡 Please install PyTorch in the backend: `pip install torch torchvision`")

    # --- Similarity Search ---
    with tab4:
        st.subheader("🔍 Image Similarity Search")
        
        if not images:
            st.warning("Please upload some images first in the Data Input section.")
            return
        
        # Initialize uploaded_ids from session state or use a placeholder
        uploaded_ids = st.session_state.get("uploaded_image_ids", [])
        
        # Silently attempt to ensure sync only if there's a clear mismatch
        if len(uploaded_ids) != len(images):
            try:
                from utils.api_client import get_current_image_ids
                backend_ids = get_current_image_ids()
                if len(backend_ids) == len(images):
                    st.session_state["uploaded_image_ids"] = backend_ids
                    uploaded_ids = backend_ids
            except:
                # If sync fails, just proceed - similarity search can still work with indices
                pass
        
        # Get available methods
        try:
            methods_info = get_similarity_methods()
            available_methods = methods_info.get("feature_methods", ["CNN", "HOG", "SIFT", "histogram"])
            available_metrics = methods_info.get("distance_metrics", ["cosine", "euclidean", "manhattan"])
        except Exception:
            available_methods = ["CNN", "HOG", "SIFT", "histogram"]
            available_metrics = ["cosine", "euclidean", "manhattan"]
        
        # Explanation of similarity scoring
        with st.expander("ℹ️ How Similarity Scores Are Calculated", expanded=False):
            st.markdown("""
            ### 🧠 **Feature Extraction Methods**
            
            **1. CNN (Convolutional Neural Network) Features:**
            - Divides image into 16×16 pixel patches
            - Calculates statistical features: mean, std, min, max, median for each patch
            - Good for: Overall image structure and texture patterns
            
            **2. HOG (Histogram of Oriented Gradients):**
            - Analyzes edge directions and gradients in image regions  
            - Parameters: orientations (default: 9), pixels per cell (8×8), cells per block (2×2)
            - Good for: Shape detection and object recognition
            
            **3. SIFT (Scale-Invariant Feature Transform):**
            - Detects distinctive keypoints and creates descriptors
            - Aggregates up to 100 keypoint descriptors into a single feature vector
            - Good for: Finding similar objects regardless of rotation/scale
            
            **4. Histogram Features:**
            - Analyzes color distribution across RGB channels
            - Creates normalized histograms with 64 bins per channel by default
            - Good for: Color-based similarity comparison
            
            ### 📏 **Distance Metrics & Score Calculation**
            
            **Cosine Similarity:**
            - Measures angle between feature vectors (0° = identical, 90° = orthogonal)
            - Score = (cosine_similarity + 1) ÷ 2  → Range: 0.0 to 1.0
            - **1.0 = identical images, 0.0 = completely different**
            
            **Euclidean Distance:**  
            - Measures straight-line distance between feature points
            - Score = exp(-distance ÷ normalization_factor)  → Range: 0.0 to 1.0
            - **1.0 = identical, lower = more different**
            
            **Manhattan Distance:**
            - Measures sum of absolute differences between features  
            - Score = exp(-distance ÷ normalization_factor)  → Range: 0.0 to 1.0
            - **1.0 = identical, lower = more different**
            
            ### 🎯 **Score Interpretation**
            - **0.9 - 1.0:** Very similar (near identical)
            - **0.7 - 0.9:** Quite similar  
            - **0.5 - 0.7:** Moderately similar
            - **0.3 - 0.5:** Somewhat different
            - **0.0 - 0.3:** Very different
            """)
            
            st.info("💡 **Tip:** Different feature methods work better for different types of images. Try multiple methods to find what works best for your data!")
        
        # Comparison mode selection
        st.markdown("### 🎯 What do you want to compare?")
        comparison_mode = st.radio(
            "Choose comparison mode:",
            ["Compare one image against all others", "Compare two specific images"],
            key="sim_comparison_mode"
        )
        
        if comparison_mode == "Compare one image against all others":
            st.markdown("**Mode: Single vs All** - Find images similar to your query image")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("##### 📷 Query Image (What to search for)")
                
                # Query image selector with preview
                query_idx = st.selectbox(
                    "Select query image:",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="sim_query_idx",
                    help="This image will be compared against all other images"
                )
                
                st.image(images[query_idx], caption=f"🔍 Query: Image {query_idx+1}", width=200)
                st.info(f"Searching for images similar to Image {query_idx+1}")
                
                # Search parameters
                st.markdown("##### ⚙️ Search Settings")
                
                # Feature method with descriptions
                method_options = {
                    "CNN": "CNN (texture patterns)",
                    "HOG": "HOG (shapes & edges)",
                    "SIFT": "SIFT (keypoint features)",
                    "histogram": "Histogram (color distribution)"
                }
                method_display = [method_options.get(method, method) for method in available_methods]
                selected_method_idx = st.selectbox("Feature Method:", range(len(available_methods)), 
                                                   format_func=lambda x: method_display[x], key="sim_method")
                feature_method = available_methods[selected_method_idx]
                
                # Distance metric with descriptions
                metric_options = {
                    "cosine": "Cosine (angle similarity)",
                    "euclidean": "Euclidean (straight distance)",
                    "manhattan": "Manhattan (grid distance)"
                }
                metric_display = [metric_options.get(metric, metric) for metric in available_metrics]
                selected_metric_idx = st.selectbox("Distance Metric:", range(len(available_metrics)),
                                                   format_func=lambda x: metric_display[x], key="sim_metric")
                distance_metric = available_metrics[selected_metric_idx]
                max_results = st.slider("Max Results:", 1, min(20, len(images)-1), 5, key="sim_max_results")
                
                # Advanced parameters
                with st.expander("🔧 Advanced Parameters"):
                    use_threshold = st.checkbox("Use similarity threshold", key="sim_use_threshold")
                    threshold = st.slider("Threshold:", 0.0, 1.0, 0.5, 0.01, key="sim_threshold") if use_threshold else None
                    
                    if feature_method == "HOG":
                        st.markdown("**HOG Parameters:**")
                        hog_orientations = st.slider("Orientations:", 1, 32, 9, key="sim_hog_orient")
                    else:
                        hog_orientations = None
                    
                    if feature_method == "histogram":
                        st.markdown("**Histogram Parameters:**")
                        hist_bins = st.slider("Bins:", 8, 256, 64, key="sim_hist_bins")
                    else:
                        hist_bins = None
            
            with col2:
                st.markdown("##### 📊 Search Results")
                
                # Performance optimization
                if st.button("🚀 Precompute Features (Recommended)", key="sim_precompute"):
                    with st.spinner("Precomputing features for faster searches..."):
                        try:
                            result = precompute_similarity_features(feature_method=feature_method)
                            st.success(f"✅ {result.get('message', 'Features precomputed successfully!')}")
                        except Exception as e:
                            st.error(f"❌ Precompute failed: {e}")
                
                # Main search button
                if st.button("🔍 Find Similar Images", type="primary", key="sim_search_btn", use_container_width=True):
                    with st.spinner(f"Searching for images similar to Image {query_idx+1}..."):
                        try:
                            # Build search parameters
                            search_params = {
                                "query_image_index": query_idx,
                                "feature_method": feature_method,
                                "distance_metric": distance_metric,
                                "max_results": max_results,
                            }
                            
                            if use_threshold and threshold is not None:
                                search_params["threshold"] = threshold
                            if feature_method == "HOG" and hog_orientations:
                                search_params["hog_orientations"] = hog_orientations
                            if feature_method == "histogram" and hist_bins:
                                search_params["hist_bins"] = hist_bins
                            
                            # Perform search
                            results = similarity_search(**search_params)
                            similar_images = results.get("similar_images", [])
                            
                            if not similar_images:
                                st.warning("No similar images found. Try adjusting parameters or threshold.")
                            else:
                                st.success(f"🎉 Found {len(similar_images)} similar images!")
                                
                                # Build image_id to index mapping
                                id_to_index = {img_id: idx for idx, img_id in enumerate(st.session_state.uploaded_image_ids)}
                                # Display results
                                for i, img_data in enumerate(similar_images[:6]):
                                    image_id = img_data["image_id"]
                                    similarity_score = img_data["similarity_score"]
                                    image_idx = id_to_index.get(image_id)
                                    result_col1, result_col2 = st.columns([1, 2])
                                    with result_col1:
                                        if image_idx is not None and image_idx < len(images):
                                            st.image(images[image_idx], width=80)
                                        else:
                                            st.write("📷")
                                    with result_col2:
                                        if image_idx is not None:
                                            st.write(f"**Image {image_idx + 1}** ({image_id})")
                                        else:
                                            st.write(f"**{image_id}**")
                                        st.metric("Similarity", f"{similarity_score:.3f}", help="Higher = more similar")
                                    if i < len(similar_images) - 1:
                                        st.markdown("---")
                                
                                # Detailed results
                                with st.expander(f"📋 All {len(similar_images)} Results"):
                                    import pandas as pd
                                    df = pd.DataFrame(similar_images)
                                    df['similarity_score'] = df['similarity_score'].round(4)
                                    df['distance'] = df['distance'].round(4)
                                    st.dataframe(df, use_container_width=True)
                        
                        except Exception as e:
                            error_msg = str(e)
                            if "list index out of range" in error_msg:
                                st.error("❌ Image synchronization error detected.")
                                st.info("💡 The images and their IDs are not properly synchronized. Try re-uploading your images.")
                                st.info("🔄 Or refresh the page and upload images again.")
                            elif "422" in error_msg or "Unprocessable Entity" in error_msg:
                                st.error("❌ Invalid request parameters. Please check your search settings.")
                                st.info("💡 Try adjusting feature method parameters or contact support.")
                            elif "400" in error_msg or "query_image_index" in error_msg:
                                st.error("❌ Invalid query image selection.")
                                st.info("💡 Make sure you have uploaded images and selected a valid query image.")
                            elif "404" in error_msg:
                                st.error("❌ Similarity search service not available.")
                                st.info("💡 Check if the backend server is running on http://localhost:8000")
                            elif "500" in error_msg:
                                st.error("❌ Internal server error occurred.")
                                st.info("💡 Try precomputing features first or check backend logs.")
                            else:
                                st.error(f"❌ Search failed: {error_msg}")
                                st.info("💡 Please try again or check your connection to the backend.")
        
        else:  # Pairwise comparison
            st.markdown("**Mode: Pairwise Comparison** - Compare two specific images directly")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown("##### 📷 First Image")
                img1_idx = st.selectbox(
                    "Select first image:",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="sim_img1_idx"
                )
                st.image(images[img1_idx], caption=f"Image {img1_idx+1}", width=150)
            
            with col2:
                st.markdown("##### 📷 Second Image")
                available_indices = [i for i in range(len(images)) if i != img1_idx]
                if available_indices:
                    img2_idx = st.selectbox(
                        "Select second image:",
                        options=available_indices,
                        format_func=lambda x: f"Image {x+1}",
                        key="sim_img2_idx"
                    )
                    st.image(images[img2_idx], caption=f"Image {img2_idx+1}", width=150)
                else:
                    st.warning("Need at least 2 images for pairwise comparison")
                    img2_idx = None
            
            with col3:
                st.markdown("##### ⚙️ Comparison Settings")
                
                # Feature method with descriptions for pairwise
                method_options = {
                    "CNN": "CNN (texture patterns)",
                    "HOG": "HOG (shapes & edges)", 
                    "SIFT": "SIFT (keypoint features)",
                    "histogram": "Histogram (color distribution)"
                }
                method_display = [method_options.get(method, method) for method in available_methods]
                selected_method_pair_idx = st.selectbox("Feature Method:", range(len(available_methods)),
                                                        format_func=lambda x: method_display[x], key="sim_method_pair")
                feature_method_pair = available_methods[selected_method_pair_idx]
                
                # Distance metric with descriptions for pairwise
                metric_options = {
                    "cosine": "Cosine (angle similarity)",
                    "euclidean": "Euclidean (straight distance)",
                    "manhattan": "Manhattan (grid distance)"
                }
                metric_display = [metric_options.get(metric, metric) for metric in available_metrics]
                selected_metric_pair_idx = st.selectbox("Distance Metric:", range(len(available_metrics)),
                                                        format_func=lambda x: metric_display[x], key="sim_metric_pair") 
                distance_metric_pair = available_metrics[selected_metric_pair_idx]
                
                if img2_idx is not None and st.button("🔍 Compare Images", type="primary", key="sim_compare_btn"):
                    with st.spinner(f"Comparing Image {img1_idx+1} vs Image {img2_idx+1}..."):
                        try:
                            # Use the first image as query and search for the second
                            search_params = {
                                "query_image_index": img1_idx,
                                "feature_method": feature_method_pair,
                                "distance_metric": distance_metric_pair,
                                "max_results": len(images),
                            }
                            
                            results = similarity_search(**search_params)
                            similar_images = results.get("similar_images", [])
                            
                            # Find the second image in results (IDs should now be synchronized)
                            uploaded_ids = st.session_state.get("uploaded_image_ids", [])
                            if img2_idx < len(uploaded_ids):
                                img2_id = uploaded_ids[img2_idx]
                                similarity_result = next((img for img in similar_images if img["image_id"] == img2_id), None)
                                
                                if similarity_result:
                                    similarity_score = similarity_result["similarity_score"]
                                    st.metric("🎯 Similarity Score", f"{similarity_score:.4f}", help="0 = completely different, 1 = identical")
                                    
                                    if similarity_score > 0.8:
                                        st.success(f"🎉 Very similar images! (Score: {similarity_score:.3f})")
                                    elif similarity_score > 0.6:
                                        st.info(f"🤔 Moderately similar images. (Score: {similarity_score:.3f})")
                                    else:
                                        st.warning(f"🤷 Images are quite different. (Score: {similarity_score:.3f})")
                                else:
                                    st.error("❌ Could not find the second image in similarity results.")
                                    st.info("💡 This might be a backend issue. Try refreshing or re-uploading images.")
                            else:
                                st.info("🔄 Images are being synchronized. Please try the comparison again.")
                        
                        except Exception as e:
                            error_msg = str(e)
                            if "list index out of range" in error_msg:
                                st.error("❌ Image synchronization error detected.")
                                st.info("💡 The images and their IDs are not properly synchronized. Try re-uploading your images.")
                                st.info("🔄 Or refresh the page and upload images again.")
                            elif "422" in error_msg or "Unprocessable Entity" in error_msg:
                                st.error("❌ Invalid comparison parameters.")
                                st.info("💡 Try adjusting feature method parameters or contact support.")
                            elif "400" in error_msg or "query_image_index" in error_msg:
                                st.error("❌ Invalid image selection for comparison.")
                                st.info("💡 Make sure you have selected two different valid images.")
                            elif "404" in error_msg:
                                st.error("❌ Similarity search service not available.")
                                st.info("💡 Check if the backend server is running on http://localhost:8000")
                            elif "500" in error_msg:
                                st.error("❌ Internal server error during comparison.")
                                st.info("💡 Try precomputing features first or check backend logs.")
                            else:
                                st.error(f"❌ Comparison failed: {error_msg}")
                                st.info("💡 Please try again or check your connection to the backend.")
        
        # Help section
        st.markdown("---")
        with st.expander("ℹ️ How to Use Similarity Search"):
            st.markdown("""
            **🎯 Comparison Modes:**
            - **Single vs All**: Select one query image and find all similar images in your dataset
            - **Pairwise**: Compare exactly two images to see how similar they are
            
            **🧠 Feature Methods:**
            - **CNN**: Best general-purpose method, works well for most images
            - **HOG**: Good for shapes, objects, and structural patterns
            - **SIFT**: Best for images with distinctive keypoints, robust to rotation/scaling
            - **Histogram**: Good for color-based similarity
            
            **📊 Distance Metrics:**
            - **Cosine**: Measures angle between feature vectors (recommended)
            - **Euclidean**: Direct distance in feature space
            - **Manhattan**: City-block distance, less sensitive to outliers
            
            **💡 Tips:**
            - Use "Precompute Features" for faster searches when you have many images
            - Try different feature methods to find what works best for your images
            - Adjust similarity threshold to filter results (higher = more strict)
            """)

    # --- Haralick Texture ---
    with tab5:
        st.subheader("Haralick Texture Tools")
        st.caption(
            "Compute Haralick texture features directly from the images you uploaded above. "
            "Choose distances, angles, quantization levels, and (optionally) resize for faster or consistent results."
        )
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

        # --- Train & Predict workflow (formerly "legacy") ---
        st.markdown("##### Train & Predict from Labeled Images")
        st.caption(
            "Train a quick classifier using Haralick features:\n"
            "1) Upload training images, 2) Upload a CSV mapping filenames to labels, 3) Upload test images to classify."
        )
        st.caption(
            "CSV example:\n"
            "`filename,label`  →  `img1.jpg,classA`  `img2.jpg,classB`"
        )

        train_imgs = st.file_uploader(
            "1) Upload Training Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="har_train_imgs"
        )
        train_csv = st.file_uploader("2) Upload Training Labels (CSV)", type="csv", key="har_train_csv")
        test_imgs = st.file_uploader(
            "3) Upload Test Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="har_test_imgs"
        )

        # Keep key the same; only label changes
        if st.button("Train & Predict", key="btn_haralick"):
            labels, preds = extract_haralick_texture(
                {"train_images": train_imgs, "train_labels": train_csv, "test_images": test_imgs}
            )
            st.write("Predicted Labels:")
            for i, p in enumerate(preds):
                st.write(f"Test Image {i+1}: {p}")

    # --- Haralick Texture ---
    with tab5:
        st.subheader("Haralick Texture Tools")

        # --- GLCM Haralick for current uploaded images (table output) ---
        st.markdown("##### Analyze Texture Features (GLCM Haralick)")
        st.caption(
            "Compute Haralick texture features directly from the images you uploaded above. "
            "Choose distances, angles, quantization levels, and (optionally) resize for faster or consistent results."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            # Select images
            use_all = st.checkbox("Use all images", value=True, key="har_use_all_moved")
            if use_all:
                image_indices = list(range(len(images)))
            else:
                image_indices = st.multiselect(
                    "Select images",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="har_img_indices_moved",
                )
            # Levels
            levels = st.selectbox("Quantization levels", [16, 32, 64, 128, 256], index=4, key="har_levels_moved")
            # Distances
            distances = st.multiselect("Distances (pixels)", [1, 2, 3, 5], default=[1, 2], key="har_distances_moved")
        with col_b:
            # Angles (radians)
            angle_map = {
                "0°": 0,
                "45°": np.pi / 4,
                "90°": np.pi / 2,
                "135°": 3 * np.pi / 4,
            }
            selected_angle_names = st.multiselect(
                "Angles",
                options=list(angle_map.keys()),
                default=["0°", "90°"],
                key="har_angles_moved"
            )
            angles = [angle_map[name] for name in selected_angle_names]
            # Resize option
            resize_enabled = st.checkbox("Resize images", value=False, key="har_resize_enabled_moved")
            if resize_enabled:
                resize_width = st.number_input("Width", min_value=32, max_value=512, value=128, key="har_resize_width_moved")
                resize_height = st.number_input("Height", min_value=32, max_value=512, value=128, key="har_resize_height_moved")
                resize_dims = [resize_width, resize_height]
            else:
                resize_dims = None

        if st.button("Extract Haralick (Table)", key="btn_haralick_table_moved"):
            if image_indices and distances and angles:
                params = {
                    "image_indices": image_indices,
                    "distances": distances,
                    "angles": angles,
                    "levels": levels,
                }
                if resize_dims:
                    params["resize_dimensions"] = resize_dims

                try:
                    result = extract_haralick_features(params)
                    # 'result' should have columns, rows, feature_names, etc.
                    columns = result.get("columns", [])
                    rows = result.get("rows", [])
                    feature_names = result.get("feature_names", [])

                    if columns and rows:
                        st.success(f"Extracted {len(rows)} feature vectors with {len(columns)-1} features each.")
                        # Build DataFrame
                        import pandas as pd
                        df = pd.DataFrame(rows, columns=columns)
                        st.dataframe(df)

                        # CSV download
                        csv_bytes = _to_csv_bytes(columns, rows)
                        st.download_button(
                            "Download Haralick CSV",
                            csv_bytes,
                            file_name="haralick_features.csv",
                            mime="text/csv",
                        )

                        # Show feature names
                        if feature_names:
                            with st.expander("Feature Descriptions"):
                                for fname in feature_names:
                                    st.write(f"- {fname}")
                    else:
                        st.warning("No features extracted. Check your parameters.")
                except Exception as e:
                    st.error(f"Haralick extraction failed: {e}")
            else:
                st.error("Please select images, distances, and angles.")

        # --- Train/Predict workflow (unchanged logic, just relabeled) ---
        st.markdown("---")
        st.markdown("##### Train/Test Workflow (Upload External Images)")
        st.caption("Upload separate training images + labels, then test images for prediction.")
        col_train, col_test = st.columns(2)
        with col_train:
            train_imgs = st.file_uploader("Training Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="har_train_moved")
            train_csv = st.file_uploader("Training Labels (CSV)", type="csv", key="har_train_csv_moved")
        with col_test:
            test_imgs = st.file_uploader("Test Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="har_test_moved")

        # Keep key the same; only label changes
        if st.button("Train & Predict", key="btn_haralick_moved"):
            labels, preds = extract_haralick_texture(
                {"train_images": train_imgs, "train_labels": train_csv, "test_images": test_imgs}
            )
            st.write("Predicted Labels:")
            for i, p in enumerate(preds):
                st.write(f"Test Image {i+1}: {p}")

    # --- Co-Occurrence Texture ---
    with tab6:
        st.subheader("Co-Occurrence Texture Features")
        tex_idx = st.selectbox(
            "Select Image", options=list(range(len(images))), format_func=lambda x: f"Image {x+1}", key="tex_img_idx"
        )
        if st.button("Extract Co-Occurrence", key="btn_cooccurrence"):
            features = extract_cooccurrence_texture({"image_index": tex_idx})
            st.write("Texture Features:")
            st.write(features)

    # --- Local Binary Patterns (LBP) ---
    with tab7:
        st.subheader("Local Binary Patterns (LBP)")

        st.caption(
            "Compute LBP texture histograms. Choose parameters below and run on one, multiple, or all images. "
            "In single-image mode, a visualization of the LBP-coded image may be shown if available."
        )

        col_l, col_r = st.columns(2)
        with col_l:
            use_all_lbp = st.checkbox("Use all images", value=False, key="lbp_use_all")
            if use_all_lbp:
                image_indices_lbp = list(range(len(images)))
                st.info(f"All {len(images)} images will be processed.")
            else:
                image_indices_lbp = st.multiselect(
                    "Select images",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Image {x+1}",
                    key="lbp_img_indices",
                )

        with col_r:
            radius = st.number_input("Radius", min_value=1, max_value=16, value=2, step=1, key="lbp_radius")
            neighbors = st.selectbox("Number of Neighbors", [8, 16, 24], index=1, key="lbp_neighbors")
            method = st.selectbox("Method", ["default", "ror", "uniform", "var"], index=2, key="lbp_method")
            normalize = st.checkbox("Normalize histogram", value=True, key="lbp_normalize")

        if st.button("Compute LBP", key="btn_lbp_compute"):
            # Validate selection when not "all"
            if not use_all_lbp and not image_indices_lbp:
                st.error("Please select at least one image or enable 'Use all images'.")
            else:
                with st.spinner("Computing LBP features..."):
                    try:
                        payload = {
                            "image_indices": image_indices_lbp,
                            "use_all_images": bool(use_all_lbp),
                            "radius": int(radius),
                            "num_neighbors": int(neighbors),
                            "method": method,
                            "normalize": bool(normalize),
                        }
                        result = extract_lbp_features(payload)

                        # Two response modes supported by backend:
                        if isinstance(result, dict) and result.get("mode") == "single":
                            # Single image: show histogram values and optional visualization
                            image_id = result.get("image_id", "image")
                            bins = int(result.get("bins", 0))
                            hist = result.get("histogram", [])
                            st.success(f"LBP histogram computed for {image_id} ({bins} bins).")
                            st.write(hist)

                            # Optional LBP-coded visualization
                            lbp_img_bytes = result.get("lbp_image_bytes")
                            if lbp_img_bytes:
                                st.image(lbp_img_bytes, caption="LBP-coded image", use_column_width=False, width=400)

                            # Download CSV
                            cols = [f"bin_{i}" for i in range(len(hist))]
                            csv_bytes = _to_csv_bytes(["image_id"] + cols, [[image_id] + hist])
                            st.download_button(
                                "Download LBP CSV (single image)",
                                data=csv_bytes,
                                file_name=f"lbp_{image_id}.csv",
                                mime="text/csv",
                                key="btn_dl_lbp_single",
                            )

                        elif isinstance(result, dict) and result.get("mode") == "multi":
                            # Multi/all images: tabular output
                            cols = result.get("columns", [])
                            rows = result.get("rows", [])
                            if cols and rows:
                                st.success(f"Computed LBP histograms for {len(rows)} image(s).")
                                st.dataframe(
                                    {cols[i]: [row[i] for row in rows] for i in range(len(cols))},
                                    use_container_width=True,
                                )
                                csv_bytes = _to_csv_bytes(cols, rows)
                                st.download_button(
                                    "Download LBP CSV",
                                    data=csv_bytes,
                                    file_name="lbp_features.csv",
                                    mime="text/csv",
                                    key="btn_dl_lbp_multi",
                                )
                            else:
                                st.warning("No LBP features returned.")
                        else:
                            st.warning("Unexpected LBP response format.")
                    except Exception as e:
                        st.error(f"LBP computation failed: {e}")

    # --- NEW: Contour Extraction (dedicated tab) ---
    with tab8:
        st.subheader("Contour Extraction")
        st.caption(
            "Extract contours (closed shapes or outlines) from a binary or grayscale version of your image using "
            "OpenCV’s `findContours`. This is useful for shape analysis, counting objects, generating bounding boxes, "
            "and creating polygonal outlines."
        )
        st.markdown(
            "**How it works:** The image is converted to grayscale and thresholded to binary. "
            "Contours are then detected and (optionally) simplified using the selected approximation method."
        )

        selected_idx = st.selectbox(
            "Select Image",
            options=list(range(len(images))),
            format_func=lambda x: f"Image {x+1}",
            key="contour_img_idx",
        )
        mode = st.selectbox(
            "Contour Retrieval Mode",
            ["RETR_EXTERNAL", "RETR_LIST", "RETR_TREE", "RETR_CCOMP"],
            index=0,
            key="contour_mode",
        )
        method = st.selectbox(
            "Contour Approximation Method",
            ["CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_NONE"],
            index=0,
            key="contour_method",
        )
        min_area = st.number_input("Minimum contour area (filter small noise)", min_value=0, value=10, key="contour_min_area")
        return_bb = st.checkbox("Return bounding boxes", value=True, key="contour_return_bb")
        return_hier = st.checkbox("Return hierarchy (OpenCV format)", value=False, key="contour_return_hier")

        if st.button("Run Contour Extraction", key="btn_contour_extract"):
            with st.spinner("Extracting contours..."):
                try:
                    params = {
                        "image_index": int(selected_idx),
                        "mode": mode,
                        "method": method,
                        "min_area": int(min_area),
                        "return_bounding_boxes": bool(return_bb),
                        "return_hierarchy": bool(return_hier),
                    }
                    result = extract_contours(params)

                    # Preview overlay
                    if result.get("visualization_bytes"):
                        st.image(
                            result["visualization_bytes"],
                            caption="Contours overlay",
                            use_column_width=False,
                            width=400,
                        )

                    # Details
                    areas = result.get("areas", [])
                    bbs = result.get("bounding_boxes", [])
                    st.write(f"Detected contours: **{len(areas)}**")
                    if bbs:
                        st.write("Bounding boxes (x, y, w, h):")
                        st.write(bbs[: min(10, len(bbs))])  # show first few

                    if return_hier:
                        st.write("Hierarchy (OpenCV):")
                        st.write(result.get("hierarchy"))
                except Exception as e:
                    st.error(f"Contour extraction failed: {e}")

    # ----------------------------------------------------------------------
#   Tab 8 – SIFT & Edge Detection
# ----------------------------------------------------------------------
    with tab9:
        st.subheader("SIFT & Edge Detection")
        st.caption(
            """
            *SIFT* extracts scale‑invariant key‑points and 128‑dimensional descriptors.  
            *Edge detection* (Canny or Sobel) highlights gradients in the image.  
            Choose a single image, a custom list, or run on **all** uploaded images.
            """
        )

        # ---------- Image selection ----------
        col_sel, col_opt = st.columns([2, 1])

        with col_sel:
            # 1️⃣  Single‑image selector (default)
            single_idx = st.selectbox(
                "Select Image (single)",
                options=list(range(len(images))),
                format_func=lambda x: f"Image {x+1}",
                key="sift_single_idx",
            )

            # 2️⃣  Multi‑select (optional)
            multi_idxs = st.multiselect(
                "Or select several images",
                options=list(range(len(images))),
                format_func=lambda x: f"Image {x+1}",
                key="sift_multi_idxs",
            )

            # 3️⃣  “All images” toggle
            use_all = st.checkbox("Run on **all** images", value=False, key="sift_use_all")

        # ---------- Edge‑detection parameters ----------
        with col_opt:
            edge_method = st.radio(
                "Edge‑Detection Method",
                options=["canny", "sobel"],
                index=0,
                key="edge_method",
            )
            if edge_method == "canny":
                low_thr = st.number_input(
                    "Low threshold", min_value=0, max_value=255, value=100, key="canny_low"
                )
                high_thr = st.number_input(
                    "High threshold", min_value=0, max_value=255, value=200, key="canny_high"
                )
            else:  # sobel
                sobel_ks = st.selectbox(
                    "Sobel kernel size",
                    options=[1, 3, 5, 7],
                    index=1,
                    key="sobel_ksize",
                )

        # ---------- Helper to build the payload ----------
        def _build_payload() -> dict:
            """
            Returns a dict that matches the Pydantic model `FeatureBaseRequest`.
            Priority (top → bottom):
                1. use_all → {"all_images": True}
                2. multi_idxs → {"image_indices": [...]}
                3. single_idx → {"image_index": …}
            """
            if use_all:
                return {"all_images": True}
            if multi_idxs:
                return {"image_indices": list(multi_idxs)}
            return {"image_index": int(single_idx)}

        # ---------- Buttons ----------
        col_btn1, col_btn2 = st.columns(2)

        # ---- SIFT ----
        with col_btn1:
            if st.button("Run SIFT", key="btn_sift"):
                payload = _build_payload()
                with st.spinner("Extracting SIFT key‑points…"):
                    try:
                        result = extract_sift_features(payload)

                        # ---- Visualisation (may be None for multi‑image requests) ----
                        viz = result.get("visualization")
                        if viz:
                            st.image(
                                viz,
                                caption="SIFT key‑points (visualisation)",
                                use_container_width=True,   # <-- NEW flag (no deprecation warning)
                            )
                        else:
                            st.info("No visualisation returned (you asked for several images).")

                        # ---- Numeric descriptors (CSV download) ----
                        feats = result.get("features", [])
                        if feats:
                            st.success(f"Extracted **{len(feats)}** SIFT descriptors.")
                            cols = [f"d{i}" for i in range(128)]
                            csv_bytes = _to_csv_bytes(cols, feats)
                            st.download_button(
                                label="Download SIFT descriptors (CSV)",
                                data=csv_bytes,
                                file_name="sift_descriptors.csv",
                                mime="text/csv",
                                key="dl_sift_csv",
                            )
                        else:
                            st.warning("No SIFT descriptors were found.")
                    except Exception as e:
                        st.error(f"SIFT extraction failed: {e}")

        # ---- Edge detection ----
        with col_btn2:
            if st.button("Run Edge Detection", key="btn_edge"):
                payload = _build_payload()
                with st.spinner("Running edge detection…"):
                    try:
                        result = extract_edge_features(
                            payload,
                            method=edge_method,
                            low_thresh=low_thr if edge_method == "canny" else None,
                            high_thresh=high_thr if edge_method == "canny" else None,
                            sobel_ksize=sobel_ks if edge_method == "sobel" else None,
                        )

                        # ---- Show the FIRST edge map ----
                        edge_imgs = result.get("edge_images", [])
                        if edge_imgs:
                            st.image(
                                edge_imgs[0],
                                caption=f"{edge_method.title()} edge map",
                                use_container_width=True,
                            )
                        else:
                            st.warning("Backend returned no edge images.")

                        # ---- Generate CSV for ALL processed images ----
                        all_matrices = result.get("edges_matrices", [])
                        num_images = len(all_matrices)
                        
                        if num_images > 0:
                            try:
                                st.success(f"Processed {num_images} image{'s' if num_images > 1 else ''}.")
                                
                                # Prepare CSV data
                                csv_rows = []
                                col_names = ["image_id", "row", "col", "value"]
                                
                                for img_idx, matrix in enumerate(all_matrices):
                                    img_id = f"img_{img_idx}"
                                    for row_idx, row in enumerate(matrix):
                                        for col_idx, value in enumerate(row):
                                            csv_rows.append([
                                                img_id,
                                                row_idx,
                                                col_idx,
                                                float(value)
                                            ])
                                
                                # Generate CSV
                                csv_bytes = _to_csv_bytes(col_names, csv_rows)
                                
                                # Download button
                                st.download_button(
                                    label=f"Download edge matrices for {num_images} image{'s' if num_images > 1 else ''} (CSV)",
                                    data=csv_bytes,
                                    file_name=f"edge_matrices_{num_images}_images.csv",
                                    mime="text/csv",
                                    key=f"dl_edge_matrices_{int(time.time())}",
                                )
                                
                                # Optional: Show matrix statistics
                                if num_images == 1:
                                    matrix = all_matrices[0]
                                    st.write(f"Matrix shape: {len(matrix)} × {len(matrix[0])}")
                                else:
                                    st.write(f"Total data points: {len(csv_rows)}")
                                    
                            except Exception as e:
                                st.error(f"CSV generation failed: {str(e)}")
                                st.write("Debug info:")
                                st.write(f"Number of images: {num_images}")
                                if num_images > 0:
                                    st.write(f"First matrix shape: {len(all_matrices[0])} × {len(all_matrices[0][0])}")
                        else:
                            st.info("No gradient matrices were generated.")
                            
                    except Exception as e:
                        st.error(f"Edge detection failed: {e}")

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
