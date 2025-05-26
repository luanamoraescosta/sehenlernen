import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# --- Page Configuration ---
st.set_page_config(page_title="Sehen Lernen", layout="wide")

# --- Session State Initialization ---
default_keys = {
    "data_prepared": False,
    "data_processed": False,
    "data_evaluated": False,
    "visualization_done": False,
    "metadata": None,
    "uploaded_files": [],
    "sampling_method": None,
    "sample_size": None,
    "processing_steps": [],
    "col_mapping": {},
}
for key, value in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Custom CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@200;600&display=swap');
body {
    background-color: #092644;
    color: #F8DFD3;
    font-family: 'Titillium Web', sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'Courier New', monospace;
    font-weight: bold;
    color: #0E3B68;
}
.block-container {
    padding: 2rem;
}
.stButton > button {
    background-color: #A23667;
    color: white;
    border-radius: 0.2rem !important;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #5E1F3B;
}
.stTabs [role="tab"] {
    background: #15599E;
    color: white;
    padding: 10px;
    border-radius: 0.2rem 0.2rem 0 0;
}
.stTabs [aria-selected="true"] {
    background: #15A4BD;
    color: black;
}
div[data-testid="stSidebar"] {
    background-color: #A98DD8;
}
img {
    object-fit: contain;
    height: auto;
}

/* Azul para botões de download */
button[kind="secondary"][data-testid="baseButton-secondary"] {
    background-color: #1565C0 !important;
    color: white !important;
}
button[kind="secondary"][data-testid="baseButton-secondary"]:hover {
    background-color: #0D47A1 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Sehen Lernen")
    if st.button("Home"):
        st.session_state["active_section"] = "Home"
    if st.button("Data Input"):
        st.session_state["active_section"] = "Data Input"
    if st.button("Feature Selection"):
        st.session_state["active_section"] = "Feature Selection"
    if st.button("Statistics Analysis"):
        st.session_state["active_section"] = "Statistics Analysis"
    if st.button("Visualization"):
        st.session_state["active_section"] = "Visualization"

# --- Title ---
st.title("Sehen Lernen")

# --- Default Section ---
if "active_section" not in st.session_state:
    st.session_state["active_section"] = "Home"

# --- Helper Functions ---
@st.cache_data
def read_csv(file, delimiter, decimal_sep):
    if file.name.endswith(".csv"):
        return pd.read_csv(file, delimiter=delimiter, decimal=decimal_sep)
    else:
        return pd.read_excel(file)

# --- Function to show large image ---
def show_large_image(index):
    st.session_state["selected_image"] = index
    st.session_state["show_large_image"] = True
    with st.sidebar:
        st.subheader("Selected Image")
        image = st.session_state["images"][index]
        st.image(image, caption=f"Image {index+1}")

# --- Home Page ---
if st.session_state["active_section"] == "Home":
    st.markdown("""
    Sehen Lernen: Human and AI Comparison

    Funded by the Innovation in Teaching Foundation as part of the Freedom 2023 program

    Contact Person: Prof. Dr. Martin Langner

    Team Members: Luana Moraes Costa, Firmin Forster, M. Kipke, Alexander Eric Wilhelm, Alexander Zeckey
    """)
    st.markdown("---")
    st.header("Welcome to the Sehen Lernen Platform")
    st.write("Here you can upload, process, and analyze images.")
    
    if st.button("Start", key="start_button"):
        st.session_state["active_section"] = "Data Input"

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("About")
        st.markdown("Sehen Lernen Project")
        st.markdown("Institut für Digital Humanities")
        st.markdown("Georg-August-Universität Göttingen")
    with col2:
        st.subheader("Tool")
        st.markdown("Version 1.0")
        st.markdown("Streamlit Platform")
        st.markdown("Machine Learning Application")
    with col3:
        st.subheader("Policies")
        st.markdown("[Privacy Policy](#)")
        st.markdown("[Terms of Service](#)")
        st.markdown("[Cookie Policy](#)")
    with col4:
        st.subheader("Contact")
        st.markdown("[Contact Us](#)")
        st.markdown("[Support Email](#)")
        st.markdown("[FAQ](#)")

# --- Data Input ---
elif st.session_state["active_section"] == "Data Input":
    st.header("Data Input")

    df = None
    uploaded_files = st.file_uploader("Select image files", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    uploaded_csv = st.file_uploader("Upload Metadata File (optional)", type=["csv", "xlsx"])

    if uploaded_csv:
        try:
            st.markdown("### Metadata Configuration")
            delimiter = st.selectbox("Select delimiter", [",", ";", "\t"])
            decimal_sep = st.selectbox("Select decimal separator", [".", ","])

            df = read_csv(uploaded_csv, delimiter, decimal_sep)
            st.write("CSV Columns:")
            cols = df.columns.tolist()
            col_mapping = {}

            image_id_col = st.selectbox("Select column for Image ID", options=cols)
            col_mapping["Image ID"] = image_id_col

            for col in cols:
                if col != image_id_col:
                    st.write(f"Column: {col}")
                    data_type = st.selectbox(f"Data type for {col}", ["Numeric", "Categorical", "Datetime", "String"], key=col)
                    col_mapping[col] = data_type

            st.session_state["col_mapping"] = col_mapping
            st.success("Column mapping completed!")

        except Exception as e:
            st.error(f"Error loading CSV: {e}")

    # Sampling Methods
    sampling_method = st.selectbox("Choose a sampling method", 
                                 ["Use All Images", 
                                  "Filter by Metadata", 
                                  "Stratified Sampling by Metadata"])

    if sampling_method == "Use All Images":
        st.session_state["sample_size"] = len(uploaded_files)
        st.write(f"All {len(uploaded_files)} images will be used.")

    elif sampling_method == "Filter by Metadata":
        if df is not None and "col_mapping" in st.session_state:
            filter_cols = st.multiselect("Select columns to filter", 
                                       options=[col for col in df.columns if col != image_id_col])
            
            filter_values = {}
            for col in filter_cols:
                values = st.multiselect(f"Select values for {col}", 
                                      options=df[col].unique())
                filter_values[col] = values

            # Apply filters
            filtered_df = df.copy()
            for col, values in filter_values.items():
                filtered_df = filtered_df[filtered_df[col].isin(values)]
            
            # Get corresponding images
            sampled_image_ids = filtered_df[image_id_col].tolist()
            sampled_files = [file for file in uploaded_files if file.name.split('/')[-1] in sampled_image_ids]
            
            st.session_state["sample_size"] = len(sampled_files)
            st.write(f"After filtering: {len(sampled_files)} images")
            
        else:
            st.warning("Please upload and configure metadata first.")

    elif sampling_method == "Stratified Sampling by Metadata":
        if df is not None and "col_mapping" in st.session_state:
            st.write("### Stratified Sampling Configuration")
            
            # Get categorical columns
            categorical_cols = [col for col, dtype in st.session_state["col_mapping"].items() 
                              if dtype == "Categorical" and col != "Image ID"]
            
            if not categorical_cols:
                st.warning("No categorical columns found for stratified sampling.")
            else:
                target_col = st.selectbox("Select target column for stratification", 
                                       options=categorical_cols)
                
                sample_size = st.number_input("Desired sample size", 
                                           min_value=1, 
                                           max_value=len(uploaded_files), 
                                           value=len(uploaded_files))
                
                # Perform stratified sampling
                from sklearn.model_selection import StratifiedSampler
                sampler = StratifiedSampler(n_samples=sample_size, random_state=42)
                
                # Prepare data for sampling
                image_ids = df[image_id_col].tolist()
                target_vector = df[target_col].tolist()
                
                # Get indices of selected images
                indices = list(range(len(image_ids)))
                sampled_indices, _ = sampler.sample(indices, target_vector)
                
                # Get corresponding files
                sampled_files = [uploaded_files[i] for i in sampled_indices]
                st.session_state["sample_size"] = len(sampled_files)
                st.write(f"Sampled {len(sampled_files)} images")
                
        else:
            st.warning("Please upload and configure metadata first.")

    else:
        st.warning("Sampling method not implemented yet.")

    # Load images
    if uploaded_files:
        st.session_state["images"] = []
        for img in uploaded_files:
            image = Image.open(io.BytesIO(img.getbuffer()))
            st.session_state["images"].append(image)

        st.session_state["uploaded_files"] = uploaded_files
        st.session_state["data_prepared"] = True

        st.subheader("Preview of Uploaded Images")
        cols = st.columns(6)
        for i, img in enumerate(st.session_state["images"]):
            with cols[i % 6]:
                width = 200
                height = (img.height * width) / img.width
                thumb = img.resize((width, int(height)))
                placeholder = st.empty()
                with placeholder.container():
                    if st.button(f"Image {i+1}", key=f"img_{i+1}_button"):
                        show_large_image(i)
                    st.image(thumb, caption=f"Image {i+1}")

    st.session_state["sampling_method"] = sampling_method

    if st.button("Next: Feature Selection", key="data_input_next"):
        st.session_state["active_section"] = "Feature Selection"

# --- Feature Selection ---
elif st.session_state["active_section"] == "Feature Selection":
    st.header("Feature Selection")
    if "images" not in st.session_state:
        st.warning("Please upload images first.")
    else:
        st.info("Feature selection functions")
        
        # --- Histogram Black&White / Colour ---
with st.expander("Histogram Black&White / Colour"):
    st.write("Select image(s) and histogram type:")
    col1, col2 = st.columns(2)
    with col1:
        histogram_type = st.selectbox("Histogram Type", ["Black & White", "Colour"], key="histogram_type_select")
        process_all_images = st.checkbox("Process all images", key="process_all_images")
    
    selected_images = []
    if process_all_images:
        selected_images = st.session_state["images"]
    else:
        st.write("Select image to analyze:")
        num_columns = 6  # Número de colunas para exibir as miniaturas
        col_list = st.columns(num_columns)
        
        for i in range(min(len(st.session_state["images"]), num_columns)):
            with col_list[i]:
                if st.button(f"Image {i+1}", key=f"image_{i+1}_button"):
                    selected_images.append(st.session_state["images"][i])
    
    if st.button("Generate Histogram", key="generate_histogram_button"):
        if not selected_images:
            st.warning("Please select at least one image to analyze.")
        else:
            try:
                fig, axes = plt.subplots(nrows=len(selected_images), ncols=1, figsize=(10, 5*len(selected_images)))
                if len(selected_images) == 1:
                    axes = [axes]
                
                for i, img in enumerate(selected_images):
                    if histogram_type == "Black & White":
                        img_gray = img.convert('L')
                        hist = np.histogram(img_gray, bins=256, range=(0, 256))[0]
                        axes[i].plot(hist, color='gray')
                        axes[i].set_title(f"Grayscale Histogram - Image {i+1}")
                    else:
                        img_rgb = img.convert('RGB')
                        r, g, b = img_rgb.split()
                        hist_r = np.histogram(r, bins=256, range=(0, 256))[0]
                        hist_g = np.histogram(g, bins=256, range=(0, 256))[0]
                        hist_b = np.histogram(b, bins=256, range=(0, 256))[0]
                        axes[i].plot(hist_r, color='red', label='Red')
                        axes[i].plot(hist_g, color='green', label='Green')
                        axes[i].plot(hist_b, color='blue', label='Blue')
                        axes[i].set_title(f"Color Histogram - Image {i+1}")
                        axes[i].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating histogram: {e}")
                
        # k-means Clustering
        with st.expander("k-means Clustering"):
            st.write("Select images from the sampling results and number of clusters:")
            col1, col2 = st.columns(2)
            with col1:
                cluster_count = st.number_input("Number of Clusters", min_value=2, max_value=10, value=2, key="cluster_count_input")
                random_state = st.number_input("Random State", min_value=0, max_value=100, value=42, key="random_state_input")
            
            if st.button("Perform k-means Clustering", key="perform_kmeans_button"):
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA
                    import matplotlib.pyplot as plt
                    
                    # Extract features from the sampled images
                    images = st.session_state["sampled_files"] if "sampled_files" in st.session_state else st.session_state["images"]
                    features = []
                    for img in images:
                        img_rgb = img.convert('RGB')
                        r, g, b = img_rgb.split()
                        hist_r = np.histogram(r, bins=32)[0]
                        hist_g = np.histogram(g, bins=32)[0]
                        hist_b = np.histogram(b, bins=32)[0]
                        features.append(np.concatenate([hist_r, hist_g, hist_b]))
                    
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(features)
                    
                    # Reduce dimensionality for visualization
                    pca = PCA(n_components=2)
                    pca_features = pca.fit_transform(scaled_features)
                    
                    # Perform k-means clustering
                    kmeans = KMeans(n_clusters=cluster_count, random_state=random_state)
                    cluster_labels = kmeans.fit_predict(pca_features)
                    
                    # Visualize clusters
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i in range(cluster_count):
                        ax.scatter(pca_features[cluster_labels == i, 0], pca_features[cluster_labels == i, 1], 
                                  label=f"Cluster {i}", alpha=0.7, s=100)
                    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                              marker='*', s=200, c='red', label='Centroids')
                    ax.set_xlabel('Principal Component 1')
                    ax.set_ylabel('Principal Component 2')
                    ax.set_title('k-means Clustering')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Display cluster assignments
                    st.write("Cluster Assignments:")
                    for i, img in enumerate(images):
                        st.write(f"Image {i+1} assigned to cluster {cluster_labels[i]}")
                except Exception as e:
                    st.error(f"Error performing k-means clustering: {e}")
        
        # Extract Shape Features
        with st.expander("Extract Shape Features"):
            st.write("Select image and feature extraction method:")
            col1, col2 = st.columns(2)
            with col1:
                shape_method = st.selectbox("Method", ["HOG", "SIFT", "FAST"], key="shape_method_select")
            with col2:
                selected_image_index = st.selectbox("Select Image", 
                                                  options=list(range(len(st.session_state["images"]))), 
                                                  key="shape_image_selector")
            
            if st.button("Extract Shape Features", key="extract_shape_features_button"):
                img = st.session_state["images"][selected_image_index]
                if shape_method == "HOG":
                    try:
                        from skimage import exposure
                        from skimage.feature import hog
                        from skimage.transform import resize
                        
                        img_gray = img.convert('L')
                        img_resized = resize(img_gray, (128, 64))  # HOG requires fixed size
                        fd, hog_image = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                                         cells_per_block=(2, 2), visualize=True)
                        
                        st.write("HOG Features:")
                        st.write(fd)
                        fig, ax = plt.subplots()
                        ax.imshow(hog_image, cmap='gray')
                        ax.set_title("HOG Visualization")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error extracting HOG features: {e}")
                elif shape_method == "SIFT":
                    try:
                        from skimage.feature import sift
                        from skimage.transform import resize
                        import cv2
                        
                        img_gray = img.convert('L')
                        img_resized = resize(img_gray, (256, 256))  # SIFT works better with larger images
                        img_resized = (img_resized * 255).astype('uint8')
                        
                        sift = cv2.SIFT_create()
                        kp, des = sift.detectAndCompute(img_resized, None)
                        
                        st.write("SIFT Features:")
                        st.write(des)
                        fig, ax = plt.subplots()
                        ax.imshow(cv2.drawKeypoints(img_resized, kp, None, color=(0, 255, 0)))
                        ax.set_title("SIFT Keypoints")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error extracting SIFT features: {e}")
                elif shape_method == "FAST":
                    try:
                        from skimage.feature import fast
                        import cv2
                        
                        img_gray = img.convert('L')
                        img_array = np.array(img_gray)
                        
                        kp = fast(img_array, threshold=30, nonmax_suppression=True)
                        
                        st.write("FAST Features:")
                        st.write(kp)
                        fig, ax = plt.subplots()
                        ax.imshow(cv2.drawKeypoints(img_array, kp, None, color=(0, 255, 0)))
                        ax.set_title("FAST Keypoints")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error extracting FAST features: {e}")
        
        # Extract Texture Features Haralick
        with st.expander("Extract Texture Features Haralick"):
            st.write("Upload labeled training images and test images:")
            col1, col2 = st.columns(2)
            with col1:
                train_images = st.file_uploader("Training Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"], key="train_images_uploader")
                train_labels_csv = st.file_uploader("Training Labels CSV", type=["csv"], key="train_labels_csv_uploader")
            with col2:
                test_images = st.file_uploader("Test Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"], key="test_images_uploader")
            
            if st.button("Extract Haralick Texture Features", key="extract_haralick_button"):
                try:
                    from skimage.feature import greycomatrix, greycoprops
                    import numpy as np
                    
                    # Load training images and labels
                    train_images_list = []
                    for img_file in train_images:
                        img = Image.open(io.BytesIO(img_file.getbuffer()))
                        img_gray = img.convert('L')
                        train_images_list.append(img_gray)
                    
                    # Read labels from CSV
                    df_labels = pd.read_csv(train_labels_csv)
                    labels = df_labels.iloc[:, 0].tolist()  # Assuming labels are in the first column
                    
                    # Extract Haralick features from training images
                    X_train = []
                    for img in train_images_list:
                        gm = greycomatrix(img, distances=[1], angles=[0], levels=256,
                                       symmetric=True, normed=True)
                        props = greycoprops(gm, 'contrast')
                        X_train.append(props[0][0].flatten())
                    
                    # Train a classifier
                    from sklearn.ensemble import RandomForestClassifier
                    clf = RandomForestClassifier(random_state=42)
                    clf.fit(X_train, labels)
                    
                    # Process test images
                    test_images_list = []
                    for img_file in test_images:
                        img = Image.open(io.BytesIO(img_file.getbuffer()))
                        img_gray = img.convert('L')
                        test_images_list.append(img_gray)
                    
                    # Extract features from test images
                    X_test = []
                    for img in test_images_list:
                        gm = greycomatrix(img, distances=[1], angles=[0], levels=256,
                                       symmetric=True, normed=True)
                        props = greycoprops(gm, 'contrast')
                        X_test.append(props[0][0].flatten())
                    
                    # Predict labels for test images
                    predicted_labels = clf.predict(X_test)
                    
                    st.write("Predicted Labels:")
                    for i, label in enumerate(predicted_labels):
                        st.write(f"Test Image {i+1}: {label}")
                except Exception as e:
                    st.error(f"Error in Haralick texture feature extraction: {e}")
       
        # Extract Texture Features Co-Occurence
        with st.expander("Extract Texture Features Co-Occurence"):
            st.write("Select image for texture feature extraction:")
            selected_image_index = st.selectbox("Select Image", 
                                              options=list(range(len(st.session_state["images"]))), 
                                              key="texture_image_selector")
            
            if st.button("Extract Co-Occurence Texture Features", key="extract_texture_button"):
                img = st.session_state["images"][selected_image_index]
                try:
                    from skimage.feature import greycomatrix, greycoprops
                    import numpy as np
                    
                    img_gray = img.convert('L')
                    gm = greycomatrix(img_gray, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                    levels=256, symmetric=True, normed=True)
                    
                    # Calculate properties
                    contrast = greycoprops(gm, 'contrast')
                    dissimilarity = greycoprops(gm, 'dissimilarity')
                    homogeneity = greycoprops(gm, 'homogeneity')
                    energy = greycoprops(gm, 'energy')
                    correlation = greycoprops(gm, 'correlation')
                    
                    features = np.array([contrast.mean(), dissimilarity.mean(), homogeneity.mean(),
                                      energy.mean(), correlation.mean()])
                    
                    st.write("Texture Features:")
                    st.write(features)
                except Exception as e:
                    st.error(f"Error extracting co-occurence texture features: {e}")
        
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Download Result", key="feature_download_result"):
                pass
        with col2:
            if st.button("Download Settings", key="feature_download_settings"):
                pass

        if st.button("Next: Statistical Analysis", key="feature_next"):
            st.session_state["active_section"] = "Statistics Analysis"

            
# --- Statistics Analysis ---
elif st.session_state["active_section"] == "Statistics Analysis":
    st.header("Statistics Analysis")
    if "embeddings" not in st.session_state:
        st.warning("Please process images first.")
    else:
        st.info("Statistics analysis placeholder")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Download Result", key="stats_download_result"):
                pass
        with col2:
            if st.button("Download Settings", key="stats_download_settings"):
                pass

        if st.button("Next: Visualization", key="stats_next"):
            st.session_state["active_section"] = "Visualization"

# --- Visualization ---
elif st.session_state["active_section"] == "Visualization":
    st.header("Visualization")
    if "reduced" not in st.session_state and "labels" not in st.session_state:
        st.warning("Please run the evaluation step to see visualizations.")
    else:
        st.info("Visualization placeholder")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Download Result", key="viz_download_result"):
                pass
        with col2:
            if st.button("Download Settings", key="viz_download_settings"):
                pass

    if st.button("Previous: Statistical Analysis", key="viz_previous"):
        st.session_state["active_section"] = "Statistics Analysis"
