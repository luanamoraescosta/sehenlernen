import streamlit as st

from components.sidebar import render_sidebar
from components.home import render_home
from components.data_input import render_data_input
from components.feature_selection import render_feature_selection
from components.stats_analysis import render_stats_analysis
from components.visualization import render_visualization

# --- Page Configuration ---
st.set_page_config(page_title="Sehen Lernen", layout="wide")

# --- Initialize Session State ---
if "active_section" not in st.session_state:
    st.session_state["active_section"] = "Home"

# --- Sidebar Navigation ---
render_sidebar()

# --- Main Content Rendering ---
section = st.session_state["active_section"]
if section == "Home":
    render_home()
elif section == "Data Input":
    render_data_input()
elif section == "Feature Selection":
    render_feature_selection()
elif section == "Statistics Analysis":
    render_stats_analysis()
elif section == "Visualization":
    render_visualization()
else:
    st.error(f"Unknown section: {section}")
