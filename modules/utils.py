# modules/utils.py

import streamlit as st

def update_console(message):
    """Update the console with new messages."""
    st.session_state.console += f"{message}\n"