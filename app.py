import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt
import io
from PIL import Image

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

    sampling_method = st.selectbox("Choose a sampling method", ["Use All Images", "Filter by Metadata", "Stratified Metadata"])

    if sampling_method == "Stratified Metadata":
        if uploaded_files:
            percent_to_sample = st.number_input("Number of images to sample", min_value=1, max_value=100, value=10, step=1)
            st.session_state["sample_size"] = int(len(uploaded_files) * (percent_to_sample / 100))
        else:
            st.warning("Please upload image files first.")

    elif sampling_method == "Filter by Metadata":
        if df is not None:
            if st.session_state["col_mapping"]:
                filtered_cols = []
                for col in st.session_state["col_mapping"]:
                    if col != "Image ID":
                        filtered_values = st.multiselect(f"Values for {col}", options=df[col].unique())
                        filtered_cols.append((col, filtered_values))

                filtered_df = df.copy()
                for col, values in filtered_cols:
                    filtered_df = filtered_df[filtered_df[col].isin(values)]

                st.session_state["sample_size"] = len(filtered_df)
            else:
                st.warning("Please upload and configure metadata first.")
        else:
            st.warning("Please upload a metadata file first.")
    else:
        st.session_state["sample_size"] = None

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
        st.info("Feature selection placeholder")
        
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
