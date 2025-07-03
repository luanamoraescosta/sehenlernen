import streamlit as st

# Visualization page component
def render_visualization():
    st.header("Visualization")

    # Ensure reduced embeddings and labels available
    if not st.session_state.get("reduced_features") or not st.session_state.get("feature_labels"):
        st.warning("Please run statistical analysis first.")
        return

    st.info("Visualization placeholder – integrate plotting of reduced embeddings, confusion matrices, etc.")

    # Download and navigation controls
    col1, col2, col3 = st.columns([2,1,1])
    with col2:
        if st.button("Download Result", key="viz_download_result"):
            pass
    with col3:
        if st.button("Download Settings", key="viz_download_settings"):
            pass

    if st.button("Previous: Statistical Analysis", key="viz_prev"):
        st.session_state["active_section"] = "Statistics Analysis"
