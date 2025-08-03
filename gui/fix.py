import streamlit as st
import sys
import os
from pathlib import Path

# Page config FIRST
st.set_page_config(
    page_title="ODE Master Generator",
    page_icon="🔬",
    layout="wide"
)

# Setup paths BEFORE any project imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Debug info
with st.expander("🔍 Debug Info"):
    st.write(f"Current dir: {current_dir}")
    st.write(f"Parent dir: {parent_dir}")
    st.write(f"sys.path[0]: {sys.path[0]}")
    
    # Check if pipeline exists
    pipeline_path = parent_dir / "pipeline"
    st.write(f"Pipeline exists: {pipeline_path.exists()}")
    if pipeline_path.exists():
        st.write(f"Files in pipeline: {list(pipeline_path.glob('*.py'))}")

# NOW do imports
try:
    st.write("Attempting imports...")
    from pipeline.generator import ODEDatasetGenerator as ODEGenerator
    from verification.verifier import ODEVerifier
    st.success("✅ Imports successful!")
    MODULES_AVAILABLE = True
    
    # Import the interface
    import standalone_complete
    # Reload to ensure it uses the updated path
    import importlib
    importlib.reload(standalone_complete)
    from standalone_complete import StandaloneODEInterface
    
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error(f"Type: {type(e).__name__}")
    import traceback
    st.code(traceback.format_exc())
    MODULES_AVAILABLE = False

# Run the app
if MODULES_AVAILABLE:
    try:
        app = StandaloneODEInterface()
        app.run()
    except Exception as e:
        st.error(f"Error running app: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.error("Cannot run due to import errors")
