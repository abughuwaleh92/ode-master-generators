"""
Main app entry point with proper import handling for Streamlit Cloud
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="ODE Master Generator",
    page_icon="🔬",
    layout="wide"
)

# Fix imports for Streamlit Cloud
def fix_imports():
    """Add parent directory to path for imports"""
    current_dir = Path(__file__).parent.absolute()
    parent_dir = current_dir.parent
    
    # Add parent to path if not already there
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Debug info (remove after it works)
    with st.expander("🔍 Debug Info (Remove when working)"):
        st.write(f"Current file: {__file__}")
        st.write(f"Current dir: {current_dir}")
        st.write(f"Parent dir: {parent_dir}")
        st.write(f"Python path: {sys.path[:3]}")
        
        # Check if files exist
        pipeline_exists = (parent_dir / "pipeline" / "generator.py").exists()
        st.write(f"pipeline/generator.py exists: {pipeline_exists}")
        
        if pipeline_exists:
            st.success("✅ Found pipeline/generator.py")
        else:
            st.error("❌ Cannot find pipeline/generator.py")
            # List what's in parent directory
            st.write("Files in parent:", list(parent_dir.glob("*")))

# Fix imports before trying to import modules
fix_imports()

# Now try imports
try:
    from pipeline.generator import ODEGenerator
    from verification.verifier import ODEVerifier
    from analyze_dataset import DatasetAnalyzer
    
    # If we get here, imports worked!
    st.sidebar.success("✅ All modules loaded successfully!")
    MODULES_AVAILABLE = True
    
    # Import and run your main interface
    from standalone_complete import StandaloneODEInterface
    
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Please check that all files are present in your repository")
    MODULES_AVAILABLE = False
    
    # Show helpful info
    st.info("""
    ### Troubleshooting:
    1. Check that all Python files are committed to GitHub
    2. Make sure there are no typos in import statements
    3. Verify the file structure matches the imports
    """)

# Run the app
if name == "__main__":
    if MODULES_AVAILABLE:
        app = StandaloneODEInterface()
        app.run()
    else:
        st.error("Cannot run app due to missing modules")
