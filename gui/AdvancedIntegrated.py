"""
ODE Master Generator - Production GUI
Real data only - No mock/demo data
Configured for Railway deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
import time
from pathlib import Path
import sympy as sp
from io import StringIO
import base64
import zipfile
import tempfile
from typing import Dict, List, Optional, Any, Tuple
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="ODE Master Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Environment configuration for Railway
API_URL = os.getenv('ODE_API_URL', 'https://your-api.railway.app')
API_KEY = os.getenv('ODE_API_KEY', '')

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = API_KEY
if 'api_url' not in st.session_state:
    st.session_state.api_url = API_URL
if 'generated_odes' not in st.session_state:
    st.session_state.generated_odes = []
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

class ODEProductionGUI:
    """Production GUI for ODE Master Generator - Real data only"""
    
    def __init__(self):
        self.api_headers = {'X-API-Key': st.session_state.api_key}
        
    def main(self):
        """Main application entry point"""
        st.title("🔬 ODE Master Generator")
        st.markdown("### Production System - Connected to Railway API")
        
        # Check API connection on startup
        if not st.session_state.api_connected:
            self.check_and_setup_api()
        
        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            
            # API connection status
            if st.session_state.api_connected:
                st.success(f"✅ Connected to API")
                st.caption(f"Endpoint: {st.session_state.api_url}")
            else:
                st.error("❌ API Not Connected")
                if st.button("Retry Connection"):
                    self.check_and_setup_api()
                    st.experimental_rerun()
            
            st.markdown("---")
            
            # Only show navigation if API is connected
            if st.session_state.api_connected:
                page = st.radio(
                    "Select Module",
                    ["🏠 Dashboard", "⚡ Generate ODEs", "✓ Verify ODEs", 
                     "📊 Analyze Dataset", "🤖 ML Pipeline", "📈 System Monitor",
                     "🔍 ODE Explorer", "⚙️ Settings"]
                )
            else:
                page = "⚙️ Settings"
        
        # Route to appropriate page
        if not st.session_state.api_connected and page != "⚙️ Settings":
            st.error("Please configure API connection in Settings first.")
            page = "⚙️ Settings"
            
        if page == "🏠 Dashboard":
            self.show_dashboard()
        elif page == "⚡ Generate ODEs":
            self.show_generation_page()
        elif page == "✓ Verify ODEs":
            self.show_verification_page()
        elif page == "📊 Analyze Dataset":
            self.show_analysis_page()
        elif page == "🤖 ML Pipeline":
            self.show_ml_page()
        elif page == "📈 System Monitor":
            self.show_monitor_page()
        elif page == "🔍 ODE Explorer":
            self.show_explorer_page()
        elif page == "⚙️ Settings":
            self.show_settings_page()
    
    def check_and_setup_api(self):
        """Check API connection and setup"""
        try:
            # Try to connect to API
            response = requests.get(
                f"{st.session_state.api_url}/health",
                headers=self.api_headers,
                timeout=5
            )
            
            if response.status_code == 200:
                st.session_state.api_connected = True
                health_data = response.json()
                logger.info(f"API connected successfully: {health_data}")
            else:
                st.session_state.api_connected = False
                logger.error(f"API connection failed with status: {response.status_code}")
        except Exception as e:
            st.session_state.api_connected = False
            logger.error(f"API connection error: {str(e)}")
    
    def show_dashboard(self):
        """Dashboard with real statistics from API"""
        st.header("📊 System Dashboard")
        
        # Fetch real statistics
        with st.spinner("Loading dashboard data..."):
            try:
                stats_response = requests.get(
                    f"{st.session_state.api_url}/api/v1/stats",
                    headers=self.api_headers,
                    timeout=10
                )
                
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                else:
                    st.error(f"Failed to fetch statistics: {stats_response.status_code}")
                    return
                    
            except Exception as e:
                st.error(f"Error fetching statistics: {str(e)}")
                return
        
        # Display real metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Generated",
                stats.get('total_generated', 0),
                delta=f"Active: {stats.get('active_jobs', 0)}"
            )
        
        with col2:
            st.metric(
                "Total Verified",
                stats.get('total_verified', 0)
            )
        
        with col3:
            st.metric(
                "Available Generators",
                stats.get('available_generators', 0)
            )
        
        with col4:
            st.metric(
                "Available Functions", 
                stats.get('available_functions', 0)
            )
        
        st.markdown("---")
        
        # System Status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖥️ API Status")
            status_data = {
                "Status": stats.get('status', 'Unknown'),
                "Redis": "✅ Connected" if stats.get('redis_available', False) else "❌ Disconnected",
                "Generators": "✅ Loaded" if stats.get('generators_available', False) else "❌ Not loaded",
                "Uptime": f"{stats.get('uptime', 0):.1f} seconds" if 'uptime' in stats else "N/A"
            }
            
            for key, value in status_data.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.subheader("📊 Job Statistics")
            
            if 'job_statistics' in stats:
                job_stats = stats['job_statistics']
                
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Completed', 'Failed', 'Running', 'Pending'],
                    values=[
                        job_stats.get('completed', 0),
                        job_stats.get('failed', 0),
                        job_stats.get('running', 0),
                        job_stats.get('pending', 0)
                    ],
                    marker_colors=['#28a745', '#dc3545', '#ffc107', '#6c757d']
                )])
                
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent Jobs
        st.subheader("📋 Recent Jobs")
        
        try:
            jobs_response = requests.get(
                f"{st.session_state.api_url}/api/v1/jobs?limit=10",
                headers=self.api_headers,
                timeout=10
            )
            
            if jobs_response.status_code == 200:
                recent_jobs = jobs_response.json()
                
                if recent_jobs:
                    job_data = []
                    for job in recent_jobs:
                        job_data.append({
                            'Job ID': job['job_id'][:8] + '...',
                            'Status': job['status'],
                            'Progress': f"{job.get('progress', 0):.0f}%",
                            'Created': job.get('created_at', '')[:19],
                            'Type': job.get('metadata', {}).get('type', 'generation')
                        })
                    
                    df = pd.DataFrame(job_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No recent jobs found")
            else:
                st.error(f"Failed to fetch jobs: {jobs_response.status_code}")
                
        except Exception as e:
            st.error(f"Error fetching recent jobs: {str(e)}")
    
    def show_generation_page(self):
        """ODE Generation page"""
        st.header("⚡ ODE Generation")
        
        # Fetch available generators and functions
        generators = self.get_available_generators()
        functions = self.get_available_functions()
        
        if not generators or not functions:
            st.error("Failed to load generators or functions from API")
            return
        
        tab1, tab2, tab3 = st.tabs(["Single Generation", "Batch Generation", "Stream Generation"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                generator = st.selectbox("Select Generator", generators)
                function = st.selectbox("Select Function", functions)
                
                st.subheader("Parameters")
                params = {}
                params['alpha'] = st.slider("Alpha (α)", 0.0, 2.0, 1.0, 0.1)
                params['beta'] = st.slider("Beta (β)", 0.1, 2.0, 1.0, 0.1)
                params['M'] = st.slider("M", -1.0, 1.0, 0.0, 0.1)
                
                if generator.startswith('N'):
                    params['q'] = st.slider("q (power)", 2, 5, 2)
                    params['v'] = st.slider("v (power)", 2, 5, 3)
                
                if generator in ['L4', 'N6']:
                    params['a'] = st.slider("a (pantograph)", 2, 5, 2)
                
                verify = st.checkbox("Verify solution", value=True)
                
                if st.button("🚀 Generate ODE", key="gen_single"):
                    self.generate_single_ode(generator, function, params, verify)
            
            with col2:
                if st.session_state.generated_odes:
                    latest_ode = st.session_state.generated_odes[-1]
                    
                    st.subheader("Generated ODE")
                    
                    # Display equation
                    if 'ode_latex' in latest_ode:
                        st.markdown("**Equation:**")
                        st.latex(latest_ode['ode_latex'])
                    elif 'ode' in latest_ode:
                        st.markdown("**Equation:**")
                        st.code(latest_ode['ode'])
                    
                    # Display solution
                    if 'solution_latex' in latest_ode:
                        st.markdown("**Solution:**")
                        st.latex(latest_ode['solution_latex'])
                    elif 'solution' in latest_ode:
                        st.markdown("**Solution:**")
                        st.code(latest_ode['solution'])
                    
                    # Verification status
                    if latest_ode.get('verified'):
                        confidence = latest_ode.get('verification_confidence', 0)
                        st.success(f"✅ Verified (Confidence: {confidence:.2%})")
                    else:
                        st.error("❌ Not verified")
                    
                    # Properties
                    with st.expander("ODE Properties"):
                        props = latest_ode.get('properties', {})
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Complexity", latest_ode.get('complexity', 'N/A'))
                            st.metric("Operations", props.get('operation_count', 'N/A'))
                        with col2:
                            st.metric("Atoms", props.get('atom_count', 'N/A'))
                            st.metric("Symbols", props.get('symbol_count', 'N/A'))
                    
                    # Download button
                    ode_json = json.dumps(latest_ode, indent=2)
                    st.download_button(
                        "📥 Download ODE JSON",
                        data=ode_json,
                        file_name=f"ode_{latest_ode.get('id', 'unknown')}.json",
                        mime="application/json"
                    )
        
        with tab2:
            st.subheader("Batch Generation")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_generators = st.multiselect(
                    "Select Generators", 
                    generators, 
                    default=generators[:min(3, len(generators))]
                )
                
                selected_functions = st.multiselect(
                    "Select Functions", 
                    functions, 
                    default=functions[:min(3, len(functions))]
                )
                
                samples_per_combo = st.number_input(
                    "Samples per combination", 
                    min_value=1, 
                    max_value=100, 
                    value=5
                )
                
                total_odes = len(selected_generators) * len(selected_functions) * samples_per_combo
                st.info(f"This will generate {total_odes} ODEs")
            
            with col2:
                st.markdown("### Options")
                verify_batch = st.checkbox("Verify all ODEs", value=True)
                
            if st.button("🚀 Start Batch Generation", key="gen_batch"):
                if selected_generators and selected_functions:
                    self.run_batch_generation(
                        selected_generators, 
                        selected_functions, 
                        samples_per_combo,
                        verify_batch
                    )
                else:
                    st.error("Please select at least one generator and function")
        
        with tab3:
            st.subheader("Real-time Stream Generation")
            
            stream_generator = st.selectbox("Generator", generators, key="stream_gen")
            stream_function = st.selectbox("Function", functions, key="stream_func")
            stream_count = st.slider("Number of ODEs", 5, 50, 10)
            
            if st.button("🌊 Start Streaming", key="start_stream"):
                container = st.empty()
                self.stream_odes(stream_generator, stream_function, stream_count, container)
    
    def show_verification_page(self):
        """ODE Verification page"""
        st.header("✓ ODE Verification")
        
        tab1, tab2 = st.tabs(["Manual Verification", "Batch Verification"])
        
        with tab1:
            st.subheader("Verify Custom ODE")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ode_input = st.text_area(
                    "Enter ODE equation",
                    value="y''(x) + y(x) = sin(x)",
                    height=100,
                    help="Use SymPy syntax: y(x) for function, y'(x) for first derivative"
                )
                
                solution_input = st.text_area(
                    "Enter solution",
                    value="C1*cos(x) + C2*sin(x) - x*cos(x)/2",
                    height=100,
                    help="Use C1, C2 for arbitrary constants"
                )
                
                method = st.selectbox(
                    "Verification method",
                    ["substitution", "numerical", "checkodesol"]
                )
                
                if st.button("🔍 Verify", key="verify_manual"):
                    self.verify_ode_manual(ode_input, solution_input, method)
            
            with col2:
                st.markdown("### Verification Guide")
                st.info("""
                **Syntax Examples:**
                - Second order: `y''(x) + 2*y'(x) + y(x) = exp(x)`
                - Nonlinear: `(y'(x))**2 + y(x) = x**2`
                - Pantograph: `y''(x) + y(x/2) = 0`
                
                **Functions:** sin, cos, exp, log, sqrt
                **Constants:** pi, e
                """)
        
        with tab2:
            st.subheader("Batch Verification")
            
            uploaded_file = st.file_uploader(
                "Upload ODE dataset (JSONL)",
                type=['jsonl', 'json'],
                help="Each line should be a JSON object with 'ode_symbolic' and 'solution_symbolic' fields"
            )
            
            if uploaded_file:
                st.success(f"Uploaded: {uploaded_file.name}")
                
                # Parse file to count ODEs
                try:
                    content = uploaded_file.read().decode('utf-8')
                    lines = content.strip().split('\n')
                    ode_count = len([l for l in lines if l.strip()])
                    st.info(f"Found {ode_count} ODEs to verify")
                    
                    uploaded_file.seek(0)  # Reset file pointer
                except:
                    st.error("Failed to parse uploaded file")
                    return
                
                col1, col2 = st.columns(2)
                
                with col1:
                    batch_method = st.selectbox(
                        "Verification method",
                        ["substitution", "numerical", "all"],
                        key="batch_method"
                    )
                
                with col2:
                    include_failed = st.checkbox(
                        "Include failed verifications in output",
                        value=True
                    )
                
                if st.button("🔍 Start Batch Verification", key="verify_batch"):
                    self.verify_batch(uploaded_file, batch_method, include_failed)
    
    def show_analysis_page(self):
        """Dataset Analysis page"""
        st.header("📊 Dataset Analysis")
        
        # Request analysis job
        st.subheader("Analyze ODE Dataset")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["summary", "generators", "functions", "complexity", "patterns"]
        )
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload dataset for analysis (optional)",
            type=['jsonl', 'json'],
            help="Leave empty to analyze server's current dataset"
        )
        
        if st.button("📊 Start Analysis"):
            with st.spinner("Running analysis..."):
                self.run_analysis(analysis_type, uploaded_file)
        
        # Display existing analysis results
        st.markdown("---")
        st.subheader("Previous Analysis Results")
        
        # Fetch completed analysis jobs
        try:
            jobs_response = requests.get(
                f"{st.session_state.api_url}/api/v1/jobs?status=completed&limit=20",
                headers=self.api_headers
            )
            
            if jobs_response.status_code == 200:
                jobs = jobs_response.json()
                analysis_jobs = [j for j in jobs if j.get('metadata', {}).get('type') == 'analysis']
                
                if analysis_jobs:
                    for job in analysis_jobs[:5]:
                        with st.expander(f"Analysis {job['job_id'][:8]} - {job['created_at'][:19]}"):
                            if 'results' in job and job['results']:
                                self.display_analysis_results(job['results'])
                            else:
                                st.info("No results available")
                else:
                    st.info("No previous analysis results found")
        except Exception as e:
            st.error(f"Failed to fetch analysis results: {str(e)}")
    
    def show_ml_page(self):
        """Machine Learning Pipeline page"""
        st.header("🤖 Machine Learning Pipeline")
        
        # Check available models
        models = self.get_available_models()
        
        tab1, tab2, tab3 = st.tabs(["Model Training", "Generate with ML", "Model Management"])
        
        with tab1:
            st.subheader("Train New Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Model Type",
                    ["pattern_net", "transformer", "vae", "language_model"]
                )
                
                dataset_source = st.radio(
                    "Dataset Source",
                    ["Use server dataset", "Upload custom dataset"]
                )
                
                if dataset_source == "Upload custom dataset":
                    training_file = st.file_uploader(
                        "Upload training dataset",
                        type=['jsonl', 'json']
                    )
                else:
                    training_file = None
                
                st.markdown("### Training Parameters")
                epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50)
                batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
                learning_rate = st.number_input(
                    "Learning rate", 
                    min_value=0.00001, 
                    max_value=0.1, 
                    value=0.001, 
                    format="%.5f"
                )
                
            with col2:
                st.markdown("### Model Configuration")
                
                config = {}
                if model_type == "pattern_net":
                    config['hidden_dims'] = st.text_input(
                        "Hidden dimensions", 
                        value="256,128,64"
                    )
                    config['dropout_rate'] = st.slider(
                        "Dropout rate", 
                        0.0, 0.5, 0.2
                    )
                elif model_type == "transformer":
                    config['n_heads'] = st.number_input(
                        "Attention heads", 
                        min_value=4, 
                        max_value=16, 
                        value=8
                    )
                    config['n_layers'] = st.number_input(
                        "Transformer layers", 
                        min_value=2, 
                        max_value=12, 
                        value=6
                    )
                elif model_type == "vae":
                    config['latent_dim'] = st.number_input(
                        "Latent dimension", 
                        min_value=16, 
                        max_value=256, 
                        value=64
                    )
                
                early_stopping = st.checkbox("Early stopping", value=True)
                config['early_stopping'] = early_stopping
            
            if st.button("🎯 Start Training", key="start_training"):
                self.start_ml_training(
                    model_type, epochs, batch_size, 
                    learning_rate, config, training_file
                )
        
        with tab2:
            st.subheader("Generate ODEs with ML Model")
            
            if models:
                selected_model = st.selectbox("Select Model", models)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    n_samples = st.number_input(
                        "Number of samples", 
                        min_value=1, 
                        max_value=1000, 
                        value=10
                    )
                    
                    temperature = st.slider(
                        "Temperature", 
                        min_value=0.1, 
                        max_value=2.0, 
                        value=0.8, 
                        step=0.1,
                        help="Lower = more conservative, Higher = more creative"
                    )
                    
                    # Constraints
                    with st.expander("Generation Constraints"):
                        generators = self.get_available_generators()
                        functions = self.get_available_functions()
                        
                        target_generator = st.selectbox(
                            "Target generator",
                            ["Any"] + generators
                        )
                        
                        target_function = st.selectbox(
                            "Target function",
                            ["Any"] + functions
                        )
                        
                        complexity_range = st.slider(
                            "Complexity range",
                            0, 1000, (50, 500)
                        )
                
                with col2:
                    st.markdown("### Generation Options")
                    
                    verify_generated = st.checkbox(
                        "Verify generated ODEs", 
                        value=True
                    )
                    
                    unique_only = st.checkbox(
                        "Unique ODEs only", 
                        value=True
                    )
                    
                if st.button("🎨 Generate with ML", key="ml_generate"):
                    self.generate_ml_odes(
                        selected_model, n_samples, temperature,
                        target_generator if target_generator != "Any" else None,
                        target_function if target_function != "Any" else None,
                        complexity_range,
                        verify_generated,
                        unique_only
                    )
            else:
                st.info("No trained models available. Train a model first.")
        
        with tab3:
            st.subheader("Model Management")
            
            if models:
                for model in models:
                    with st.expander(f"Model: {model['name']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Type:** {model.get('metadata', {}).get('model_type', 'Unknown')}")
                            st.write(f"**Created:** {model.get('created', 'Unknown')}")
                        
                        with col2:
                            st.write(f"**Size:** {model.get('size', 0) / 1024 / 1024:.2f} MB")
                            st.write(f"**Epochs:** {model.get('metadata', {}).get('epochs', 'Unknown')}")
                        
                        with col3:
                            if st.button(f"Delete", key=f"delete_{model['name']}"):
                                st.warning("Model deletion not implemented yet")
                            
                            if st.button(f"Download", key=f"download_{model['name']}"):
                                st.info("Model download not implemented yet")
            else:
                st.info("No models found")
    
    def show_monitor_page(self):
        """System monitoring page"""
        st.header("📈 System Monitor")
        
        # Refresh controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            auto_refresh = st.checkbox("Auto-refresh", value=False)
        
        with col2:
            refresh_interval = st.selectbox(
                "Interval (seconds)",
                [5, 10, 30, 60],
                index=1
            )
        
        with col3:
            if st.button("🔄 Refresh Now"):
                st.experimental_rerun()
        
        # Fetch current metrics
        try:
            stats_response = requests.get(
                f"{st.session_state.api_url}/api/v1/stats",
                headers=self.api_headers
            )
            
            metrics_response = requests.get(
                f"{st.session_state.api_url}/metrics",
                headers=self.api_headers
            )
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Active Jobs", stats.get('active_jobs', 0))
                
                with col2:
                    st.metric("Total Generated", stats.get('total_generated', 0))
                
                with col3:
                    st.metric("Cache Size", stats.get('cache_size', 0))
                
                with col4:
                    uptime = stats.get('uptime', 0)
                    st.metric("Uptime", f"{uptime/3600:.1f}h" if uptime > 3600 else f"{uptime/60:.1f}m")
                
                # Job distribution
                if 'job_statistics' in stats:
                    st.subheader("Job Statistics")
                    
                    job_stats = stats['job_statistics']
                    
                    # Create charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Job status pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=['Completed', 'Failed', 'Running', 'Pending'],
                            values=[
                                job_stats.get('completed', 0),
                                job_stats.get('failed', 0),
                                job_stats.get('running', 0),
                                job_stats.get('pending', 0)
                            ],
                            hole=.3
                        )])
                        fig.update_layout(title="Job Status Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Success rate gauge
                        total_jobs = job_stats.get('total', 1)
                        success_rate = (job_stats.get('completed', 0) / total_jobs * 100) if total_jobs > 0 else 0
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = success_rate,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Success Rate %"},
                            delta = {'reference': 90},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkgreen"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                
                # Prometheus metrics if available
                if metrics_response.status_code == 200:
                    st.subheader("Performance Metrics")
                    metrics_text = metrics_response.text
                    
                    # Parse and display key metrics
                    if "ode_generation_total" in metrics_text:
                        st.code(metrics_text[:500] + "...", language="text")
            
            else:
                st.error(f"Failed to fetch statistics: {stats_response.status_code}")
                
        except Exception as e:
            st.error(f"Error fetching metrics: {str(e)}")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.experimental_rerun()
    
    def show_explorer_page(self):
        """ODE Explorer - browse generated ODEs"""
        st.header("🔍 ODE Explorer")
        
        # Fetch recent ODEs from completed jobs
        st.subheader("Recent Generated ODEs")
        
        # Pagination
        page = st.number_input("Page", min_value=1, value=1)
        per_page = st.selectbox("Items per page", [10, 25, 50], index=0)
        
        # Fetch completed generation jobs
        try:
            offset = (page - 1) * per_page
            jobs_response = requests.get(
                f"{st.session_state.api_url}/api/v1/jobs?status=completed&limit={per_page}&offset={offset}",
                headers=self.api_headers
            )
            
            if jobs_response.status_code == 200:
                jobs = jobs_response.json()
                
                # Filter for generation jobs with results
                ode_results = []
                for job in jobs:
                    if job.get('results') and isinstance(job['results'], list):
                        for result in job['results']:
                            if 'ode' in result:
                                ode_results.append(result)
                
                if ode_results:
                    st.success(f"Found {len(ode_results)} ODEs")
                    
                    # Display ODEs
                    for i, ode in enumerate(ode_results):
                        with st.expander(
                            f"ODE {i+1}: {ode.get('generator', 'Unknown')} - "
                            f"{ode.get('function', 'Unknown')}"
                        ):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Equation:**")
                                if 'ode_latex' in ode:
                                    st.latex(ode['ode_latex'])
                                else:
                                    st.code(ode.get('ode', 'N/A'))
                                
                                if 'solution_latex' in ode:
                                    st.markdown("**Solution:**")
                                    st.latex(ode['solution_latex'])
                                elif 'solution' in ode:
                                    st.markdown("**Solution:**")
                                    st.code(ode['solution'])
                            
                            with col2:
                                st.markdown("**Properties:**")
                                props = {
                                    'ID': ode.get('id', 'N/A'),
                                    'Verified': '✅' if ode.get('verified') else '❌',
                                    'Complexity': ode.get('complexity', 'N/A'),
                                    'Generator': ode.get('generator', 'N/A'),
                                    'Function': ode.get('function', 'N/A')
                                }
                                
                                for key, value in props.items():
                                    st.write(f"**{key}:** {value}")
                                
                                # Download individual ODE
                                ode_json = json.dumps(ode, indent=2)
                                st.download_button(
                                    "📥 Download",
                                    data=ode_json,
                                    file_name=f"ode_{ode.get('id', i)}.json",
                                    mime="application/json",
                                    key=f"download_ode_{i}"
                                )
                else:
                    st.info("No ODEs found. Generate some ODEs first!")
            else:
                st.error(f"Failed to fetch ODEs: {jobs_response.status_code}")
                
        except Exception as e:
            st.error(f"Error fetching ODEs: {str(e)}")
    
    def show_settings_page(self):
        """Settings and configuration page"""
        st.header("⚙️ Settings")
        
        tab1, tab2 = st.tabs(["API Configuration", "About"])
        
        with tab1:
            st.subheader("API Configuration")
            
            # Show current configuration
            st.info(f"Current API URL: {st.session_state.api_url}")
            
            # API configuration form
            with st.form("api_config"):
                api_url = st.text_input(
                    "API URL",
                    value=st.session_state.api_url,
                    help="Your Railway API endpoint (e.g., https://your-app.railway.app)"
                )
                
                api_key = st.text_input(
                    "API Key",
                    value=st.session_state.api_key,
                    type="password",
                    help="Your API authentication key"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    test_connection = st.form_submit_button("🔌 Test Connection")
                
                with col2:
                    save_settings = st.form_submit_button("💾 Save Settings")
            
            if test_connection:
                with st.spinner("Testing connection..."):
                    try:
                        test_headers = {'X-API-Key': api_key}
                        response = requests.get(
                            f"{api_url}/health",
                            headers=test_headers,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            health_data = response.json()
                            st.success("✅ Connection successful!")
                            
                            # Display API info
                            st.json(health_data)
                        else:
                            st.error(f"❌ Connection failed: HTTP {response.status_code}")
                            try:
                                error_data = response.json()
                                st.error(f"Error: {error_data}")
                            except:
                                st.error(f"Response: {response.text}")
                                
                    except requests.exceptions.Timeout:
                        st.error("❌ Connection timeout - check if API is running")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Connection error - check API URL")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            if save_settings:
                st.session_state.api_url = api_url
                st.session_state.api_key = api_key
                self.api_headers = {'X-API-Key': api_key}
                
                # Test the new settings
                self.check_and_setup_api()
                
                if st.session_state.api_connected:
                    st.success("✅ Settings saved and connected!")
                    time.sleep(1)
                    st.experimental_rerun()
                else:
                    st.error("❌ Settings saved but connection failed")
            
            # Environment variables info
            st.markdown("---")
            st.subheader("Environment Variables")
            st.info("""
            For Railway deployment, set these environment variables:
            - `ODE_API_URL`: Your API endpoint
            - `ODE_API_KEY`: Your API key
            
            This will automatically configure the connection on startup.
            """)
        
        with tab2:
            st.subheader("About ODE Master Generator")
            
            st.markdown("""
            ### Version 2.0.0 - Production
            
            This is the production GUI for the ODE Master Generator system, 
            connected to your Railway-deployed API.
            
            ### Features
            - ✅ Real-time ODE generation with exact solutions
            - ✅ Multiple verification methods
            - ✅ Comprehensive dataset analysis
            - ✅ Machine learning pipeline
            - ✅ System monitoring
            - ✅ Batch processing capabilities
            
            ### API Endpoints Used
            - `/health` - Connection status
            - `/api/v1/generate` - ODE generation
            - `/api/v1/verify` - Solution verification
            - `/api/v1/analyze` - Dataset analysis
            - `/api/v1/ml/*` - Machine learning operations
            - `/api/v1/jobs` - Job management
            - `/api/v1/stats` - System statistics
            
            ### Support
            For issues or questions, check the API logs in Railway dashboard.
            """)
    
    # Helper methods
    def get_available_generators(self):
        """Fetch available generators from API"""
        try:
            response = requests.get(
                f"{st.session_state.api_url}/api/v1/generators",
                headers=self.api_headers,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('all', [])
        except Exception as e:
            logger.error(f"Failed to fetch generators: {str(e)}")
        return []
    
    def get_available_functions(self):
        """Fetch available functions from API"""
        try:
            response = requests.get(
                f"{st.session_state.api_url}/api/v1/functions",
                headers=self.api_headers,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('functions', [])
        except Exception as e:
            logger.error(f"Failed to fetch functions: {str(e)}")
        return []
    
    def get_available_models(self):
        """Fetch available ML models from API"""
        try:
            response = requests.get(
                f"{st.session_state.api_url}/api/v1/models",
                headers=self.api_headers,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
        except Exception as e:
            logger.error(f"Failed to fetch models: {str(e)}")
        return []
    
    def generate_single_ode(self, generator, function, params, verify):
        """Generate a single ODE via API"""
        with st.spinner("Generating ODE..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/api/v1/generate",
                    headers=self.api_headers,
                    json={
                        "generator": generator,
                        "function": function,
                        "parameters": params,
                        "count": 1,
                        "verify": verify
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    job_id = result['job_id']
                    
                    # Poll for results
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(60):  # 60 second timeout
                        time.sleep(1)
                        progress_bar.progress(min(i / 30, 0.99))
                        
                        status_response = requests.get(
                            f"{st.session_state.api_url}/api/v1/jobs/{job_id}",
                            headers=self.api_headers
                        )
                        
                        if status_response.status_code == 200:
                            job_status = status_response.json()
                            
                            status_text.text(f"Status: {job_status['status']}")
                            
                            if job_status['status'] == 'completed':
                                progress_bar.progress(1.0)
                                odes = job_status.get('results', [])
                                if odes:
                                    st.session_state.generated_odes.append(odes[0])
                                    st.success("✅ ODE generated successfully!")
                                    st.balloons()
                                else:
                                    st.error("No ODE in results")
                                return
                                
                            elif job_status['status'] == 'failed':
                                st.error(f"Generation failed: {job_status.get('error', 'Unknown error')}")
                                return
                    
                    st.warning("Generation timed out")
                else:
                    st.error(f"API error: {response.status_code}")
                    try:
                        error_data = response.json()
                        st.error(f"Details: {error_data}")
                    except:
                        st.error(f"Response: {response.text}")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    def run_batch_generation(self, generators, functions, samples, verify):
        """Run batch ODE generation"""
        total = len(generators) * len(functions) * samples
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        completed = 0
        all_results = []
        
        for gen in generators:
            for func in functions:
                status_text.text(f"Generating: {gen} + {func} ({samples} samples)...")
                
                try:
                    response = requests.post(
                        f"{st.session_state.api_url}/api/v1/generate",
                        headers=self.api_headers,
                        json={
                            "generator": gen,
                            "function": func,
                            "count": samples,
                            "verify": verify
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        job_id = result['job_id']
                        
                        # Poll for completion
                        for _ in range(120):  # 2 minute timeout
                            time.sleep(1)
                            
                            job_response = requests.get(
                                f"{st.session_state.api_url}/api/v1/jobs/{job_id}",
                                headers=self.api_headers
                            )
                            
                            if job_response.status_code == 200:
                                job_data = job_response.json()
                                
                                if job_data['status'] == 'completed':
                                    results = job_data.get('results', [])
                                    all_results.extend(results)
                                    completed += samples
                                    progress_bar.progress(completed / total)
                                    
                                    with results_container:
                                        st.success(f"✅ {gen} + {func}: {len(results)} ODEs generated")
                                    break
                                    
                                elif job_data['status'] == 'failed':
                                    with results_container:
                                        st.error(f"❌ {gen} + {func}: Failed")
                                    break
                    
                except Exception as e:
                    with results_container:
                        st.error(f"❌ {gen} + {func}: Error - {str(e)}")
        
        status_text.text(f"Batch complete! Generated {len(all_results)} ODEs")
        
        if all_results:
            # Create downloadable file
            jsonl_content = '\n'.join(json.dumps(ode) for ode in all_results)
            
            st.download_button(
                "📥 Download All ODEs (JSONL)",
                data=jsonl_content,
                file_name=f"batch_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                mime="application/x-jsonlines"
            )
    
    def stream_odes(self, generator, function, count, container):
        """Stream ODEs in real-time"""
        try:
            # Use SSE endpoint
            response = requests.get(
                f"{st.session_state.api_url}/api/v1/stream/generate",
                headers=self.api_headers,
                params={
                    "generator": generator,
                    "function": function,
                    "count": count
                },
                stream=True,
                timeout=120
            )
            
            if response.status_code == 200:
                ode_count = 0
                
                for line in response.iter_lines():
                    if line:
                        if line.startswith(b'data: '):
                            try:
                                data = json.loads(line[6:])
                                ode_count += 1
                                
                                with container.container():
                                    st.subheader(f"ODE {ode_count}/{count}")
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        if 'ode_latex' in data:
                                            st.latex(data['ode_latex'])
                                        else:
                                            st.code(data.get('ode', ''))
                                    
                                    with col2:
                                        if data.get('verified'):
                                            st.success("✅ Verified")
                                        else:
                                            st.error("❌ Not verified")
                                    
                                    # Progress
                                    progress = data.get('progress', (ode_count/count)*100)
                                    st.progress(progress/100)
                                    
                            except json.JSONDecodeError:
                                continue
                
                container.success(f"✅ Streamed {ode_count} ODEs successfully!")
                
            else:
                container.error(f"Stream error: {response.status_code}")
                
        except Exception as e:
            container.error(f"Streaming error: {str(e)}")
    
    def verify_ode_manual(self, ode_str, solution_str, method):
        """Verify manually entered ODE"""
        with st.spinner("Verifying..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/api/v1/verify",
                    headers=self.api_headers,
                    json={
                        "ode": ode_str,
                        "solution": solution_str,
                        "method": method
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['verified']:
                        st.success(f"✅ Verified! Confidence: {result['confidence']:.2%}")
                        st.info(f"Method used: {result['method']}")
                    else:
                        st.error("❌ Verification failed")
                    
                    # Show details
                    with st.expander("Verification Details"):
                        st.json(result.get('details', {}))
                        
                else:
                    st.error(f"API error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Verification error: {str(e)}")
    
    def verify_batch(self, uploaded_file, method, include_failed):
        """Verify batch of ODEs"""
        # This would send file to API for batch verification
        st.info("Batch verification via API - Implementation pending")
        
        # For now, show what would happen
        st.code("""
        # Would send to API:
        POST /api/v1/verify/batch
        {
            "file": uploaded_file,
            "method": method,
            "include_failed": include_failed
        }
        """)
    
    def run_analysis(self, analysis_type, uploaded_file):
        """Run dataset analysis"""
        try:
            # Prepare request
            request_data = {
                "analysis_type": analysis_type
            }
            
            # If file uploaded, include it
            files = {}
            if uploaded_file:
                files['dataset'] = uploaded_file
            
            response = requests.post(
                f"{st.session_state.api_url}/api/v1/analyze",
                headers=self.api_headers,
                data=request_data,
                files=files if files else None,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                
                st.info(f"Analysis job created: {job_id}")
                st.info("Check job status in Recent Jobs or refresh this page")
                
            else:
                st.error(f"Failed to start analysis: {response.status_code}")
                
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
    
    def display_analysis_results(self, results):
        """Display analysis results"""
        if isinstance(results, dict):
            # Summary statistics
            if 'statistics' in results:
                st.subheader("Statistics")
                stats = results['statistics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total ODEs", stats.get('total_odes', 0))
                with col2:
                    st.metric("Verification Rate", f"{stats.get('verified_rate', 0):.1%}")
                with col3:
                    st.metric("Avg Complexity", f"{stats.get('avg_complexity', 0):.1f}")
            
            # Generator distribution
            if 'generator_distribution' in results:
                st.subheader("Generator Distribution")
                gen_dist = results['generator_distribution']
                
                fig = px.bar(
                    x=list(gen_dist.keys()),
                    y=list(gen_dist.values()),
                    title="ODEs by Generator"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Other results
            for key, value in results.items():
                if key not in ['statistics', 'generator_distribution']:
                    st.subheader(key.replace('_', ' ').title())
                    if isinstance(value, dict):
                        st.json(value)
                    elif isinstance(value, list):
                        st.write(value)
                    else:
                        st.write(value)
        else:
            st.json(results)
    
    def start_ml_training(self, model_type, epochs, batch_size, learning_rate, config, training_file):
        """Start ML model training"""
        with st.spinner("Starting training job..."):
            try:
                request_data = {
                    "model_type": model_type,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "config": config
                }
                
                if training_file:
                    request_data["dataset"] = training_file.name
                else:
                    request_data["dataset"] = "server_dataset"
                
                response = requests.post(
                    f"{st.session_state.api_url}/api/v1/ml/train",
                    headers=self.api_headers,
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    job_id = result['job_id']
                    
                    st.success(f"Training job started: {job_id}")
                    st.info("Monitor progress in the job queue or check back later")
                    
                    # Store job ID for tracking
                    st.session_state.training_job_id = job_id
                    
                else:
                    st.error(f"Failed to start training: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Training error: {str(e)}")
    
    def generate_ml_odes(self, model, n_samples, temperature, target_gen, target_func, 
                        complexity_range, verify, unique_only):
        """Generate ODEs using ML model"""
        with st.spinner("Generating ODEs with ML model..."):
            try:
                request_data = {
                    "model_path": model['path'],
                    "n_samples": n_samples,
                    "temperature": temperature,
                    "generator": target_gen,
                    "function": target_func,
                    "complexity_range": list(complexity_range) if complexity_range else None
                }
                
                response = requests.post(
                    f"{st.session_state.api_url}/api/v1/ml/generate",
                    headers=self.api_headers,
                    json=request_data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    job_id = result['job_id']
                    
                    # Poll for results
                    progress_bar = st.progress(0)
                    
                    for i in range(120):  # 2 minute timeout
                        time.sleep(1)
                        progress_bar.progress(min(i / 60, 0.99))
                        
                        job_response = requests.get(
                            f"{st.session_state.api_url}/api/v1/jobs/{job_id}",
                            headers=self.api_headers
                        )
                        
                        if job_response.status_code == 200:
                            job_data = job_response.json()
                            
                            if job_data['status'] == 'completed':
                                progress_bar.progress(1.0)
                                
                                results = job_data.get('results', {})
                                generated_odes = results.get('odes', [])
                                
                                st.success(f"Generated {len(generated_odes)} ODEs!")
                                
                                # Display results
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Generated", len(generated_odes))
                                with col2:
                                    st.metric("Valid", results.get('valid_count', 0))
                                with col3:
                                    st.metric("Avg Novelty", f"{results.get('avg_novelty_score', 0):.2f}")
                                
                                # Show generated ODEs
                                if generated_odes:
                                    st.subheader("Generated ODEs")
                                    for i, ode in enumerate(generated_odes[:5]):  # Show first 5
                                        with st.expander(f"ODE {i+1}"):
                                            st.code(ode.get('ode', ''))
                                            if ode.get('valid'):
                                                st.success("✅ Valid")
                                            else:
                                                st.error("❌ Invalid")
                                
                                # Download all
                                if generated_odes:
                                    jsonl_content = '\n'.join(json.dumps(ode) for ode in generated_odes)
                                    st.download_button(
                                        "📥 Download Generated ODEs",
                                        data=jsonl_content,
                                        file_name=f"ml_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                                        mime="application/x-jsonlines"
                                    )
                                
                                return
                                
                            elif job_data['status'] == 'failed':
                                st.error(f"Generation failed: {job_data.get('error', 'Unknown error')}")
                                return
                    
                    st.warning("ML generation timed out")
                    
                else:
                    st.error(f"Failed to start ML generation: {response.status_code}")
                    
            except Exception as e:
                st.error(f"ML generation error: {str(e)}")

# Main execution
if __name__ == "__main__":
    app = ODEProductionGUI()
    app.main()
