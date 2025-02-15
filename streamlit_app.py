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
    st.error(f"Failed to import required modules: {str(e)}")
    st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}") 