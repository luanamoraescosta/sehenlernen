import streamlit as st

# Sidebar navigation component
def render_sidebar():
    with st.sidebar:
        st.title("Sehen Lernen")
        # Navigation buttons
        if st.button("Home", key="nav_home"):
            st.session_state["active_section"] = "Home"
        if st.button("Data Input", key="nav_data_input"):
            st.session_state["active_section"] = "Data Input"
        if st.button("Feature Selection", key="nav_feature_selection"):
            st.session_state["active_section"] = "Feature Selection"
        if st.button("Statistics Analysis", key="nav_stats_analysis"):
            st.session_state["active_section"] = "Statistics Analysis"
        if st.button("Visualization", key="nav_visualization"):
            st.session_state["active_section"] = "Visualization"
