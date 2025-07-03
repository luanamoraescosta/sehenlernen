import streamlit as st

# Home page component
def render_home():
    st.markdown("""
    Sehen Lernen: Human and AI Comparison

    Funded by the Innovation in Teaching Foundation as part of the Freedom 2023 program

    Contact Person: Prof. Dr. Martin Langner

    Team Members: Luana Moraes Costa, Firmin Forster, M. Kipke, Alexander Eric Wilhelm, Alexander Zeckey
    """)
    st.markdown("---")
    st.header("Welcome to the Sehen Lernen Platform")
    st.write("Here you can upload, process, and analyze images.")

    if st.button("Start", key="home_start_button"):
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
