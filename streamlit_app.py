import sys
from pathlib import Path
import streamlit as st

# Configure paths
try:
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.append(str(src_path))
    
    # Import dashboard
    from dashboard.streamlit_app import main
except ImportError as e:
    st.error(f"""Failed to import required modules: {str(e)}
    Please ensure all dependencies are installed correctly.
    If the error persists, contact support.""")
    st.stop()

def run_app():
    try:
        main()
    except Exception as e:
        st.error(f"""Application error: {str(e)}
        Please try refreshing the page.
        If the error persists, contact support.""")

if __name__ == "__main__":
    run_app() 