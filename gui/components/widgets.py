"""Custom Streamlit widgets"""

import streamlit as st

def parameter_input_group():
    """Create a standard parameter input group"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alpha = st.slider("α", -5.0, 5.0, 1.0, 0.1)
    with col2:
        beta = st.slider("β", 0.1, 5.0, 1.0, 0.1)
    with col3:
        M = st.slider("M", -5.0, 5.0, 0.0, 0.1)
    
    return {"alpha": alpha, "beta": beta, "M": M}