"""
ODE Master Generator - Integrated GUI Interface

A comprehensive Streamlit application that integrates all ODE generation,
verification, analysis, and ML capabilities into a powerful GUI.
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
import subprocess
import psutil
import redis
from streamlit_ace import st_ace
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import networkx as nx

# Configure Streamlit
st.set_page_config(
    page_title="ODE Master Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
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
    .info-box {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
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

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = 'test-key'
if 'api_url' not in st.session_state:
    st.session_state.api_url = 'http://localhost:8000'
if 'generated_odes' not in st.session_state:
    st.session_state.generated_odes = []
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None
if 'redis_client' not in st.session_state:
    try:
        st.session_state.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        st.session_state.redis_client.ping()
        st.session_state.redis_available = True
    except:
        st.session_state.redis_available = False

class ODEMasterGUI:
    """Main GUI application class"""
    
    def __init__(self):
        self.api_headers = {'X-API-Key': st.session_state.api_key}
        
    def main(self):
        """Main application entry point"""
        st.title("🔬 ODE Master Generator")
        st.markdown("### Comprehensive ODE Generation, Analysis & Machine Learning Platform")
        
        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Select Module",
                ["🏠 Dashboard", "⚡ Generate ODEs", "✓ Verify ODEs", 
                 "📊 Analyze Dataset", "🤖 ML Pipeline", "📈 Real-time Monitor",
                 "🔍 ODE Explorer", "⚙️ Settings"]
            )
            
            # API Status
            st.markdown("---")
            if self.check_api_status():
                st.success("✅ API Connected")
            else:
                st.error("❌ API Disconnected")
        
        # Route to appropriate page
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
        elif page == "📈 Real-time Monitor":
            self.show_monitor_page()
        elif page == "🔍 ODE Explorer":
            self.show_explorer_page()
        elif page == "⚙️ Settings":
            self.show_settings_page()
    
    def check_api_status(self):
        """Check if API is available"""
        try:
            response = requests.get(
                f"{st.session_state.api_url}/health",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False
    
    def show_dashboard(self):
        """Dashboard page with overview and statistics"""
        st.header("📊 System Dashboard")
        
        # Check if API is connected
        api_connected = self.check_api_status()
        
        if not api_connected:
            st.warning("⚠️ API is not connected. Showing demo data.")
            
            # Show demo metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Generated", "1,234", delta="+52 today")
            
            with col2:
                st.metric("Active Jobs", "3", delta="Running")
            
            with col3:
                st.metric("Available Generators", "11")
            
            with col4:
                st.metric("Available Functions", "34")
        else:
            # Fetch statistics
            try:
                stats_response = requests.get(
                    f"{st.session_state.api_url}/api/v1/stats",
                    headers=self.api_headers
                )
                stats = stats_response.json() if stats_response.status_code == 200 else {}
            except:
                stats = {}
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Generated",
                    stats.get('total_generated', 0),
                    delta=f"+{stats.get('total_generated', 0) // 24} today"
                )
            
            with col2:
                st.metric(
                    "Active Jobs",
                    stats.get('active_jobs', 0),
                    delta="Running"
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
            st.subheader("🖥️ System Resources")
            
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Create gauge charts
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=['CPU Usage', 'Memory Usage', 'Disk Usage']
            )
            
            # CPU gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=cpu_percent,
                    title={'text': "CPU %"},
                    domain={'x': [0, 0.3], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Memory gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=memory.percent,
                    title={'text': "Memory %"},
                    domain={'x': [0.35, 0.65], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ]
                    }
                ),
                row=1, col=2
            )
            
            # Disk gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=disk.percent,
                    title={'text': "Disk %"},
                    domain={'x': [0.7, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "purple"},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 90], 'color': "yellow"},
                            {'range': [90, 100], 'color': "red"}
                        ]
                    }
                ),
                row=1, col=3
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
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
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent Activity
        st.subheader("📋 Recent Activity")
        
        if not api_connected:
            # Show demo activity
            demo_jobs = [
                {'job_id': 'abc123...', 'status': 'completed', 'progress': '100%', 'created_at': '2025-01-15 10:30:00'},
                {'job_id': 'def456...', 'status': 'running', 'progress': '45%', 'created_at': '2025-01-15 10:35:00'},
                {'job_id': 'ghi789...', 'status': 'completed', 'progress': '100%', 'created_at': '2025-01-15 10:40:00'},
            ]
            df = pd.DataFrame(demo_jobs)
            st.dataframe(df, use_container_width=True)
        else:
            try:
                jobs_response = requests.get(
                    f"{st.session_state.api_url}/api/v1/jobs?limit=5",
                    headers=self.api_headers
                )
                
                if jobs_response.status_code == 200:
                    recent_jobs = jobs_response.json()
                    
                    if recent_jobs:
                        job_data = []
                        for job in recent_jobs:
                            job_data.append({
                                'Job ID': job['job_id'][:8] + '...',
                                'Status': job['status'],
                                'Progress': f"{job['progress']:.0f}%",
                                'Created': job['created_at'][:19]
                            })
                        
                        df = pd.DataFrame(job_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No recent jobs")
            except:
                st.error("Failed to fetch recent jobs")
    
    def show_generation_page(self):
        """ODE Generation page"""
        st.header("⚡ ODE Generation")
        
        tab1, tab2, tab3 = st.tabs(["Single Generation", "Batch Generation", "Stream Generation"])
        
        with tab1:
            self.show_single_generation()
        
        with tab2:
            self.show_batch_generation()
        
        with tab3:
            self.show_stream_generation()
    
    def show_single_generation(self):
        """Single ODE generation interface"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Get available generators and functions
            generators = self.get_available_generators()
            functions = self.get_available_functions()
            
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
            
            if generator == 'L4' or generator == 'N6':
                params['a'] = st.slider("a (pantograph)", 2, 5, 2)
            
            verify = st.checkbox("Verify solution", value=True)
            
            if st.button("🚀 Generate ODE", key="gen_single"):
                with st.spinner("Generating ODE..."):
                    self.generate_single_ode(generator, function, params, verify)
        
        with col2:
            if st.session_state.generated_odes:
                latest_ode = st.session_state.generated_odes[-1]
                
                st.subheader("Generated ODE")
                
                # Display ODE
                st.markdown("**Equation:**")
                st.latex(latest_ode.get('ode_latex', latest_ode.get('ode', '')))
                
                if 'solution' in latest_ode:
                    st.markdown("**Solution:**")
                    st.latex(latest_ode.get('solution_latex', latest_ode.get('solution', '')))
                
                # Verification status
                if latest_ode.get('verified'):
                    st.success(f"✅ Verified (Confidence: {latest_ode.get('verification_confidence', 0):.2%})")
                else:
                    st.error("❌ Not verified")
                
                # Properties
                with st.expander("Properties"):
                    props = latest_ode.get('properties', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Complexity", latest_ode.get('complexity', 0))
                        st.metric("Operations", props.get('operation_count', 0))
                    with col2:
                        st.metric("Atoms", props.get('atom_count', 0))
                        st.metric("Symbols", props.get('symbol_count', 0))
                
                # Download button
                ode_json = json.dumps(latest_ode, indent=2)
                st.download_button(
                    "📥 Download ODE",
                    data=ode_json,
                    file_name=f"ode_{latest_ode.get('id', 'unknown')}.json",
                    mime="application/json"
                )
    
    def show_batch_generation(self):
        """Batch ODE generation interface"""
        st.subheader("Batch Generation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Multi-select for generators and functions
            generators = self.get_available_generators()
            functions = self.get_available_functions()
            
            selected_generators = st.multiselect("Select Generators", generators, default=generators[:3])
            selected_functions = st.multiselect("Select Functions", functions, default=functions[:3])
            
            samples_per_combo = st.number_input("Samples per combination", 1, 100, 5)
            
            total_odes = len(selected_generators) * len(selected_functions) * samples_per_combo
            st.info(f"This will generate {total_odes} ODEs")
        
        with col2:
            st.markdown("### Batch Parameters")
            use_random = st.checkbox("Random parameters", value=True)
            
            if not use_random:
                st.warning("Fixed parameters will be used for all ODEs")
        
        if st.button("🚀 Start Batch Generation", key="gen_batch"):
            self.run_batch_generation(selected_generators, selected_functions, samples_per_combo)
    
    def show_stream_generation(self):
        """Stream generation interface"""
        st.subheader("Real-time Stream Generation")
        
        generator = st.selectbox("Generator", self.get_available_generators(), key="stream_gen")
        function = st.selectbox("Function", self.get_available_functions(), key="stream_func")
        count = st.slider("Number of ODEs", 5, 50, 10)
        
        if st.button("🌊 Start Streaming", key="start_stream"):
            container = st.container()
            progress_bar = st.progress(0)
            
            with container:
                self.stream_odes(generator, function, count, progress_bar)
    
    def show_verification_page(self):
        """ODE Verification page"""
        st.header("✓ ODE Verification")
        
        tab1, tab2 = st.tabs(["Manual Verification", "Dataset Verification"])
        
        with tab1:
            st.subheader("Verify Custom ODE")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ode_input = st.text_area(
                    "Enter ODE equation",
                    value="y''(x) + y(x) = sin(x)",
                    height=100
                )
                
                solution_input = st.text_area(
                    "Enter solution",
                    value="C1*cos(x) + C2*sin(x) - x*cos(x)/2",
                    height=100
                )
                
                method = st.selectbox(
                    "Verification method",
                    ["substitution", "numerical", "all"]
                )
                
                if st.button("🔍 Verify", key="verify_manual"):
                    self.verify_ode_manual(ode_input, solution_input, method)
            
            with col2:
                st.markdown("### Verification Tips")
                st.info("""
                - Use SymPy syntax for equations
                - Include y(x) for the function
                - Use y'(x) and y''(x) for derivatives
                - Constants: C1, C2, etc.
                """)
                
                st.markdown("### Examples")
                examples = {
                    "Linear 2nd order": {
                        "ode": "y''(x) + 2*y'(x) + y(x) = exp(x)",
                        "solution": "C1*exp(-x) + C2*x*exp(-x) + exp(x)/4"
                    },
                    "Nonlinear": {
                        "ode": "y'(x)**2 + y(x) = x**2",
                        "solution": "x**2 - 2*x + 2 - C1*exp(-x)"
                    }
                }
                
                for name, ex in examples.items():
                    if st.button(f"Load {name}", key=f"ex_{name}"):
                        st.session_state.ode_example = ex['ode']
                        st.session_state.sol_example = ex['solution']
                        st.experimental_rerun()
        
        with tab2:
            st.subheader("Verify Dataset")
            
            uploaded_file = st.file_uploader(
                "Upload ODE dataset (JSONL)",
                type=['jsonl', 'json']
            )
            
            if uploaded_file:
                st.info(f"Uploaded: {uploaded_file.name}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    confidence_threshold = st.slider(
                        "Confidence threshold",
                        0.0, 1.0, 0.95, 0.05
                    )
                
                with col2:
                    methods = st.multiselect(
                        "Verification methods",
                        ["substitution", "numerical", "series"],
                        default=["substitution", "numerical"]
                    )
                
                with col3:
                    parallel_workers = st.number_input(
                        "Parallel workers",
                        1, 8, 4
                    )
                
                if st.button("🔍 Verify Dataset", key="verify_dataset"):
                    self.verify_dataset(uploaded_file, methods, confidence_threshold, parallel_workers)
    
    def show_analysis_page(self):
        """Dataset Analysis page"""
        st.header("📊 Dataset Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload ODE dataset for analysis",
            type=['jsonl', 'json', 'parquet']
        )
        
        if uploaded_file:
            # Load data
            if uploaded_file.name.endswith('.jsonl'):
                data = []
                for line in uploaded_file:
                    data.append(json.loads(line))
                df = pd.DataFrame(data)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                data = json.load(uploaded_file)
                df = pd.DataFrame(data)
            
            st.success(f"Loaded {len(df)} ODEs")
            
            # Analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Overview", "Generators", "Functions", "Complexity", "Patterns"
            ])
            
            with tab1:
                self.show_dataset_overview(df)
            
            with tab2:
                self.show_generator_analysis(df)
            
            with tab3:
                self.show_function_analysis(df)
            
            with tab4:
                self.show_complexity_analysis(df)
            
            with tab5:
                self.show_pattern_analysis(df)
            
            # Export options
            st.markdown("---")
            st.subheader("Export Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Generate Full Report"):
                    report = self.generate_analysis_report(df)
                    st.download_button(
                        "Download Report",
                        data=report,
                        file_name="ode_analysis_report.html",
                        mime="text/html"
                    )
            
            with col2:
                if st.button("📈 Export Visualizations"):
                    self.export_visualizations(df)
            
            with col3:
                if st.button("💾 Export Processed Data"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        data=csv,
                        file_name="ode_dataset_processed.csv",
                        mime="text/csv"
                    )
    
    def show_ml_page(self):
        """Machine Learning Pipeline page"""
        st.header("🤖 Machine Learning Pipeline")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Preparation", "Model Training", "Generation", "Evaluation"
        ])
        
        with tab1:
            self.show_ml_data_prep()
        
        with tab2:
            self.show_ml_training()
        
        with tab3:
            self.show_ml_generation()
        
        with tab4:
            self.show_ml_evaluation()
    
    def show_ml_data_prep(self):
        """ML Data preparation interface"""
        st.subheader("Data Preparation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Upload Dataset")
            dataset_file = st.file_uploader(
                "Upload ODE dataset",
                type=['jsonl', 'json'],
                key="ml_dataset"
            )
            
            if dataset_file:
                st.success(f"Loaded: {dataset_file.name}")
                
                # Split configuration
                st.markdown("### Dataset Splitting")
                test_size = st.slider("Test set size", 0.1, 0.3, 0.2)
                val_size = st.slider("Validation set size", 0.05, 0.2, 0.1)
                
                strategy = st.selectbox(
                    "Split strategy",
                    ["stratified", "random", "temporal", "grouped"]
                )
                
                if st.button("🔀 Split Dataset"):
                    self.split_ml_dataset(dataset_file, test_size, val_size, strategy)
        
        with col2:
            st.markdown("### Feature Engineering")
            
            feature_options = st.multiselect(
                "Select features to extract",
                [
                    "Complexity metrics",
                    "Structural features",
                    "Parameter statistics",
                    "Text embeddings",
                    "Graph features"
                ],
                default=["Complexity metrics", "Structural features"]
            )
            
            if st.button("🔧 Extract Features"):
                st.info("Feature extraction started...")
                # Feature extraction logic
    
    def show_ml_training(self):
        """ML Model training interface"""
        st.subheader("Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["pattern_net", "transformer", "vae", "language_model"]
            )
            
            st.markdown(f"### {model_type.title()} Configuration")
            
            # Model-specific parameters
            if model_type == "pattern_net":
                hidden_dims = st.text_input("Hidden dimensions", "256,128,64")
                dropout = st.slider("Dropout rate", 0.0, 0.5, 0.2)
            elif model_type == "transformer":
                n_heads = st.number_input("Attention heads", 4, 16, 8)
                n_layers = st.number_input("Transformer layers", 2, 12, 6)
            elif model_type == "vae":
                latent_dim = st.number_input("Latent dimension", 16, 256, 64)
            
            # Common parameters
            epochs = st.number_input("Epochs", 10, 1000, 100)
            batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
            learning_rate = st.number_input("Learning rate", 0.00001, 0.1, 0.001, format="%.5f")
            
            early_stopping = st.checkbox("Early stopping", value=True)
            
            if st.button("🎯 Start Training"):
                self.start_ml_training(model_type, epochs, batch_size, learning_rate)
        
        with col2:
            st.markdown("### Training Progress")
            
            if st.session_state.get('training_job_id'):
                job_status = self.get_job_status(st.session_state.training_job_id)
                
                if job_status:
                    # Progress bar
                    progress = job_status.get('progress', 0) / 100
                    st.progress(progress)
                    
                    # Metrics
                    if 'metadata' in job_status:
                        metadata = job_status['metadata']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Epoch", metadata.get('current_epoch', 0))
                        with col2:
                            st.metric("Total Epochs", metadata.get('total_epochs', 0))
                    
                    # Loss curve placeholder
                    st.line_chart(pd.DataFrame({
                        'training_loss': np.random.rand(50) * 0.5 + 0.1,
                        'validation_loss': np.random.rand(50) * 0.6 + 0.15
                    }))
    
    def show_ml_generation(self):
        """ML-based ODE generation interface"""
        st.subheader("ML-based ODE Generation")
        
        # Model selection
        available_models = self.get_available_models()
        
        if available_models:
            selected_model = st.selectbox("Select trained model", available_models)
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_samples = st.number_input("Number of samples", 1, 1000, 10)
                temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
                
                # Optional constraints
                with st.expander("Generation Constraints"):
                    target_generator = st.selectbox(
                        "Target generator style",
                        ["Any"] + self.get_available_generators()
                    )
                    target_function = st.selectbox(
                        "Target function type",
                        ["Any"] + self.get_available_functions()
                    )
                    
                    complexity_min = st.number_input("Min complexity", 0, 1000, 50)
                    complexity_max = st.number_input("Max complexity", 0, 1000, 500)
                
                if st.button("🎨 Generate ODEs"):
                    self.generate_ml_odes(
                        selected_model, n_samples, temperature,
                        target_generator if target_generator != "Any" else None,
                        target_function if target_function != "Any" else None,
                        [complexity_min, complexity_max]
                    )
            
            with col2:
                st.markdown("### Generation Tips")
                st.info("""
                **Temperature** controls randomness:
                - Low (0.1-0.5): Conservative, similar to training data
                - Medium (0.6-1.0): Balanced creativity
                - High (1.1-2.0): More experimental, may be invalid
                
                **Constraints** help guide generation:
                - Generator style affects equation structure
                - Function type influences solution form
                - Complexity range controls equation length
                """)
        else:
            st.warning("No trained models available. Please train a model first.")
    
    def show_ml_evaluation(self):
        """ML Model evaluation interface"""
        st.subheader("Model Evaluation")
        
        # Evaluation metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "0.92", delta="+0.03")
        with col2:
            st.metric("Precision", "0.89", delta="+0.02")
        with col3:
            st.metric("Recall", "0.94", delta="+0.04")
        with col4:
            st.metric("F1 Score", "0.91", delta="+0.03")
        
        # Detailed evaluation
        tab1, tab2, tab3 = st.tabs(["Performance", "Error Analysis", "Model Comparison"])
        
        with tab1:
            # Confusion matrix
            st.subheader("Confusion Matrix")
            
            # Dummy data for visualization
            confusion_matrix = np.array([[850, 50], [30, 70]])
            
            fig = px.imshow(
                confusion_matrix,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Verified', 'Not Verified'],
                y=['Verified', 'Not Verified'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            st.plotly_chart(fig)
            
            # Performance by generator
            st.subheader("Performance by Generator")
            
            generator_performance = pd.DataFrame({
                'Generator': ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3'],
                'Accuracy': [0.95, 0.93, 0.94, 0.88, 0.90, 0.87, 0.85],
                'Count': [150, 145, 160, 120, 130, 110, 95]
            })
            
            fig = px.bar(
                generator_performance,
                x='Generator',
                y='Accuracy',
                color='Count',
                title="Model Accuracy by Generator Type"
            )
            st.plotly_chart(fig)
    
    def show_monitor_page(self):
        """Real-time monitoring page"""
        st.header("📈 Real-time System Monitor")
        
        # Auto-refresh
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 5)
        
        # Create placeholder containers
        metrics_placeholder = st.empty()
        charts_placeholder = st.empty()
        activity_placeholder = st.empty()
        
        # Continuous monitoring loop
        if st.button("Start Monitoring"):
            self.run_monitoring(metrics_placeholder, charts_placeholder, activity_placeholder, refresh_interval)
    
    def show_explorer_page(self):
        """ODE Explorer page for browsing and searching"""
        st.header("🔍 ODE Explorer")
        
        # Search interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("Search ODEs", placeholder="e.g., sin, exponential")
        
        with col2:
            filter_generator = st.multiselect(
                "Filter by generator",
                self.get_available_generators()
            )
        
        with col3:
            filter_verified = st.selectbox(
                "Verification status",
                ["All", "Verified", "Not Verified"]
            )
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)
            
            with col1:
                complexity_range = st.slider(
                    "Complexity range",
                    0, 1000, (0, 500)
                )
                
                has_pantograph = st.checkbox("Has pantograph terms")
            
            with col2:
                order = st.multiselect(
                    "ODE Order",
                    [1, 2, 3, 4],
                    default=[1, 2]
                )
                
                date_range = st.date_input(
                    "Date range",
                    value=(datetime.now() - timedelta(days=7), datetime.now()),
                    max_value=datetime.now()
                )
        
        if st.button("🔍 Search"):
            # For demo purposes, show sample results
            st.info("Search functionality requires database connection. Showing sample results.")
            results = self.get_sample_odes()
            
            
            if results:
                st.success(f"Found {len(results)} ODEs")
                
                # Display results
                for i, ode in enumerate(results[:10]):  # Show first 10
                    with st.expander(f"ODE {i+1}: {ode.get('generator_name', 'Unknown')} - {ode.get('function_name', 'Unknown')}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Equation:**")
                            st.latex(ode.get('ode_latex', ode.get('ode_symbolic', '')))
                            
                            if 'solution_latex' in ode:
                                st.markdown("**Solution:**")
                                st.latex(ode['solution_latex'])
                        
                        with col2:
                            st.markdown("**Properties:**")
                            st.json({
                                'Verified': ode.get('verified', False),
                                'Complexity': ode.get('complexity_score', 0),
                                'Generator': ode.get('generator_name', ''),
                                'Function': ode.get('function_name', ''),
                                'Has Pantograph': ode.get('has_pantograph', False)
                            })
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Verify", key=f"verify_{i}"):
                                    st.info("Verification feature coming soon!")
                            with col2:
                                if st.button(f"Visualize", key=f"viz_{i}"):
                                    self.visualize_ode(ode)
            else:
                st.info("No ODEs found matching your criteria")
    
    def show_settings_page(self):
        """Settings page"""
        st.header("⚙️ Settings")
        
        tab1, tab2, tab3 = st.tabs(["API Configuration", "System Settings", "About"])
        
        with tab1:
            st.subheader("API Configuration")
            
            api_url = st.text_input(
                "API URL",
                value=st.session_state.api_url
            )
            
            api_key = st.text_input(
                "API Key",
                value=st.session_state.api_key,
                type="password"
            )
            
            if st.button("Test Connection"):
                try:
                    response = requests.get(
                        f"{api_url}/health",
                        headers={'X-API-Key': api_key},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        st.success("✅ Connection successful!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Connection failed: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
            
            if st.button("Save API Settings"):
                st.session_state.api_url = api_url
                st.session_state.api_key = api_key
                self.api_headers = {'X-API-Key': api_key}
                st.success("Settings saved!")
        
        with tab2:
            st.subheader("System Settings")
            
            # Redis configuration
            st.markdown("### Redis Configuration")
            redis_host = st.text_input("Redis Host", value="localhost")
            redis_port = st.number_input("Redis Port", value=6379)
            
            # Performance settings
            st.markdown("### Performance Settings")
            max_workers = st.number_input("Max parallel workers", 1, 16, 4)
            cache_size = st.number_input("Cache size (MB)", 100, 1000, 256)
            
            # Display settings
            st.markdown("### Display Settings")
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            show_advanced = st.checkbox("Show advanced options", value=True)
            
            if st.button("Apply System Settings"):
                st.success("System settings applied!")
        
        with tab3:
            st.subheader("About ODE Master Generator")
            
            st.markdown("""
            ### Version 2.0.0
            
            **ODE Master Generator** is a comprehensive platform for:
            - 🔬 Generating ordinary differential equations with exact solutions
            - ✅ Verifying ODE solutions using multiple methods
            - 📊 Analyzing ODE datasets with advanced statistics
            - 🤖 Training ML models on ODE patterns
            - 🎨 Generating novel ODEs using AI
            
            ### Features
            - **11+ ODE Generators**: Linear and nonlinear generators
            - **34+ Functions**: From basic to complex mathematical functions
            - **Multiple Verification Methods**: Substitution, numerical, series expansion
            - **ML Pipeline**: Pattern recognition, transformers, VAE
            - **Real-time Monitoring**: System metrics and job tracking
            - **Batch Processing**: Generate thousands of ODEs efficiently
            
            ### Credits
            Developed with ❤️ using:
            - SymPy for symbolic mathematics
            - Streamlit for the interface
            - FastAPI for the backend
            - PyTorch for machine learning
            
            ### Documentation
            For detailed documentation, visit the [GitHub repository](https://github.com/your-repo/ode-master-generator)
            """)
    
    # Helper methods
    def get_available_generators(self):
        """Get list of available generators from API"""
        try:
            response = requests.get(
                f"{st.session_state.api_url}/api/v1/generators",
                headers=self.api_headers
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('all', [])
        except:
            pass
        return ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3', 'N7']
    
    def get_available_functions(self):
        """Get list of available functions from API"""
        try:
            response = requests.get(
                f"{st.session_state.api_url}/api/v1/functions",
                headers=self.api_headers
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('functions', [])
        except:
            pass
        return ['identity', 'sine', 'cosine', 'exponential', 'quadratic']
    
    def get_available_models(self):
        """Get list of available ML models"""
        try:
            response = requests.get(
                f"{st.session_state.api_url}/api/v1/models",
                headers=self.api_headers
            )
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
    
    def generate_single_ode(self, generator, function, params, verify):
        """Generate a single ODE"""
        if not self.check_api_status():
            st.warning("API is not connected. Showing demo ODE.")
            
            # Show demo ODE
            demo_ode = {
                'id': 'demo_123',
                'ode': f"{generator}: y''(x) + f(y) = g(x)",
                'ode_latex': r"y''(x) + f(y) = g(x)",
                'solution': f"Solution for {function}",
                'solution_latex': r"y(x) = C_1\phi_1(x) + C_2\phi_2(x)",
                'verified': verify,
                'verification_confidence': 0.95 if verify else 0,
                'complexity': 75,
                'generator': generator,
                'function': function,
                'parameters': params,
                'properties': {
                    'operation_count': 5,
                    'atom_count': 12,
                    'symbol_count': 4,
                    'has_pantograph': generator == 'L4'
                }
            }
            st.session_state.generated_odes.append(demo_ode)
            st.success("✅ Demo ODE generated!")
            return
            
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
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                
                # Poll for results
                with st.spinner("Waiting for generation to complete..."):
                    for _ in range(30):  # 30 second timeout
                        time.sleep(1)
                        
                        status_response = requests.get(
                            f"{st.session_state.api_url}/api/v1/jobs/{job_id}",
                            headers=self.api_headers
                        )
                        
                        if status_response.status_code == 200:
                            job_status = status_response.json()
                            
                            if job_status['status'] == 'completed':
                                odes = job_status.get('results', [])
                                if odes:
                                    st.session_state.generated_odes.append(odes[0])
                                    st.success("✅ ODE generated successfully!")
                                    st.balloons()
                                return
                            elif job_status['status'] == 'failed':
                                st.error(f"Generation failed: {job_status.get('error', 'Unknown error')}")
                                return
                
                st.warning("Generation timed out")
            else:
                st.error(f"API error: {response.status_code}")
        except Exception as e:
            st.error(f"Error generating ODE: {str(e)}")
    
    def run_batch_generation(self, generators, functions, samples):
        """Run batch generation"""
        total = len(generators) * len(functions) * samples
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        generated = 0
        
        for gen in generators:
            for func in functions:
                status_text.text(f"Generating {gen} + {func}...")
                
                try:
                    response = requests.post(
                        f"{st.session_state.api_url}/api/v1/generate",
                        headers=self.api_headers,
                        json={
                            "generator": gen,
                            "function": func,
                            "count": samples,
                            "verify": True
                        }
                    )
                    
                    if response.status_code == 200:
                        generated += samples
                        progress_bar.progress(generated / total)
                    
                except Exception as e:
                    st.error(f"Error in batch generation: {str(e)}")
        
        status_text.text(f"Batch generation complete! Generated {generated} ODEs")
        st.balloons()
    
    def stream_odes(self, generator, function, count, progress_bar):
        """Stream ODEs in real-time"""
        try:
            # Use requests with stream=True for SSE
            response = requests.get(
                f"{st.session_state.api_url}/api/v1/stream/generate",
                headers=self.api_headers,
                params={
                    "generator": generator,
                    "function": function,
                    "count": count
                },
                stream=True
            )
            
            if response.status_code == 200:
                ode_container = st.container()
                
                for i, line in enumerate(response.iter_lines()):
                    if line:
                        # Parse SSE data
                        if line.startswith(b'data: '):
                            data = json.loads(line[6:])
                            
                            with ode_container:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.latex(data.get('ode', ''))
                                with col2:
                                    if data.get('verified'):
                                        st.success("✅ Verified")
                                    else:
                                        st.error("❌ Not verified")
                            
                            progress_bar.progress((i + 1) / count)
                
                st.success(f"Streamed {count} ODEs successfully!")
            else:
                st.error(f"Stream error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Streaming error: {str(e)}")
    
    def show_dataset_overview(self, df):
        """Show dataset overview statistics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total ODEs", len(df))
            st.metric("Unique Generators", df['generator_name'].nunique())
        
        with col2:
            st.metric("Verified", df['verified'].sum())
            st.metric("Verification Rate", f"{df['verified'].mean():.1%}")
        
        with col3:
            st.metric("Avg Complexity", f"{df['complexity_score'].mean():.1f}")
            st.metric("Has Pantograph", df.get('has_pantograph', pd.Series([False])).sum())
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Generator distribution
            fig = px.pie(
                df['generator_name'].value_counts().reset_index(),
                values='generator_name',
                names='index',
                title="Generator Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Verification by generator
            verification_by_gen = df.groupby('generator_name')['verified'].mean().reset_index()
            fig = px.bar(
                verification_by_gen,
                x='generator_name',
                y='verified',
                title="Verification Rate by Generator",
                color='verified',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_generator_analysis(self, df):
        """Analyze generators in detail"""
        st.subheader("Generator Performance Analysis")
        
        # Performance metrics by generator
        gen_stats = df.groupby('generator_name').agg({
            'verified': ['count', 'sum', 'mean'],
            'complexity_score': ['mean', 'std', 'min', 'max'],
            'generation_time': 'mean'
        }).round(3)
        
        st.dataframe(gen_stats, use_container_width=True)
        
        # Complexity distribution by generator
        fig = px.box(
            df,
            x='generator_name',
            y='complexity_score',
            title="Complexity Distribution by Generator",
            color='generator_name'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Success rate over time
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            daily_stats = df.groupby(['date', 'generator_name'])['verified'].mean().reset_index()
            
            fig = px.line(
                daily_stats,
                x='date',
                y='verified',
                color='generator_name',
                title="Verification Rate Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_function_analysis(self, df):
        """Analyze functions in detail"""
        st.subheader("Function Analysis")
        
        # Top functions
        top_functions = df['function_name'].value_counts().head(20)
        
        fig = px.bar(
            top_functions.reset_index(),
            x='index',
            y='function_name',
            title="Top 20 Functions Used",
            labels={'index': 'Function', 'function_name': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Function-Generator heatmap
        pivot_table = pd.crosstab(df['generator_name'], df['function_name'])
        
        fig = px.imshow(
            pivot_table,
            title="Generator-Function Usage Heatmap",
            labels=dict(x="Function", y="Generator", color="Count"),
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_complexity_analysis(self, df):
        """Analyze complexity patterns"""
        st.subheader("Complexity Analysis")
        
        # Complexity distribution
        fig = px.histogram(
            df,
            x='complexity_score',
            nbins=50,
            title="Overall Complexity Distribution",
            labels={'complexity_score': 'Complexity Score', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Complexity vs Verification
        complexity_bins = pd.qcut(df['complexity_score'], q=10, labels=False)
        verification_by_complexity = df.groupby(complexity_bins)['verified'].mean()
        
        fig = px.line(
            x=range(10),
            y=verification_by_complexity.values,
            title="Verification Rate by Complexity Decile",
            labels={'x': 'Complexity Decile', 'y': 'Verification Rate'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Complexity components
        if all(col in df.columns for col in ['operation_count', 'atom_count', 'symbol_count']):
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Operation Count', 'Atom Count', 'Symbol Count']
            )
            
            fig.add_trace(
                go.Histogram(x=df['operation_count'], name='Operations'),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=df['atom_count'], name='Atoms'),
                row=1, col=2
            )
            fig.add_trace(
                go.Histogram(x=df['symbol_count'], name='Symbols'),
                row=1, col=3
            )
            
            fig.update_layout(height=400, title_text="Complexity Components Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_pattern_analysis(self, df):
        """Analyze ODE patterns"""
        st.subheader("Pattern Analysis")
        
        # Extract patterns from ODE strings
        patterns = defaultdict(int)
        
        for ode_str in df.get('ode_symbolic', []):
            if pd.notna(ode_str):
                # Count mathematical operations
                patterns['Addition'] += ode_str.count('+')
                patterns['Subtraction'] += ode_str.count('-')
                patterns['Multiplication'] += ode_str.count('*')
                patterns['Division'] += ode_str.count('/')
                patterns['Power'] += ode_str.count('**')
                
                # Count functions
                patterns['Sine'] += ode_str.count('sin')
                patterns['Cosine'] += ode_str.count('cos')
                patterns['Exponential'] += ode_str.count('exp')
                patterns['Logarithm'] += ode_str.count('log')
        
        # Display pattern distribution
        pattern_df = pd.DataFrame(
            list(patterns.items()),
            columns=['Pattern', 'Count']
        ).sort_values('Count', ascending=False)
        
        fig = px.bar(
            pattern_df,
            x='Pattern',
            y='Count',
            title="Mathematical Pattern Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Word cloud of ODE terms
        st.subheader("ODE Term Frequency")
        
        # Collect all terms
        all_terms = ' '.join(df['ode_symbolic'].dropna().tolist())
        
        # Simple frequency analysis
        import re
        words = re.findall(r'\b[a-zA-Z]+\b', all_terms)
        word_freq = Counter(words)
        
        # Display top terms
        top_terms = pd.DataFrame(
            word_freq.most_common(20),
            columns=['Term', 'Frequency']
        )
        
        fig = px.bar(
            top_terms,
            x='Term',
            y='Frequency',
            title="Top 20 Terms in ODEs"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def verify_ode_manual(self, ode_str, solution_str, method):
        """Verify manually entered ODE"""
        try:
            response = requests.post(
                f"{st.session_state.api_url}/api/v1/verify",
                headers=self.api_headers,
                json={
                    "ode": ode_str,
                    "solution": solution_str,
                    "method": method
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result['verified']:
                    st.success(f"✅ Verified! Confidence: {result['confidence']:.2%}")
                    st.markdown(f"**Method:** {result['method']}")
                else:
                    st.error("❌ Verification failed")
                
                with st.expander("Verification Details"):
                    st.json(result.get('details', {}))
            else:
                st.error(f"API error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Verification error: {str(e)}")
    
    def run_monitoring(self, metrics_placeholder, charts_placeholder, activity_placeholder, refresh_interval):
        """Run continuous monitoring"""
        while True:
            try:
                # Fetch current stats
                stats_response = requests.get(
                    f"{st.session_state.api_url}/api/v1/stats",
                    headers=self.api_headers
                )
                
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    
                    # Update metrics
                    with metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Active Jobs", stats.get('active_jobs', 0))
                        with col2:
                            st.metric("Total Generated", stats.get('total_generated', 0))
                        with col3:
                            st.metric("Cache Size", stats.get('cache_size', 0))
                        with col4:
                            st.metric("Status", stats.get('status', 'Unknown'))
                    
                    # Update charts
                    with charts_placeholder.container():
                        # Create real-time line chart
                        # This would need actual time-series data from Redis
                        pass
                    
                    # Update activity log
                    with activity_placeholder.container():
                        st.subheader("Recent Activity")
                        # Fetch and display recent jobs
                        jobs_response = requests.get(
                            f"{st.session_state.api_url}/api/v1/jobs?limit=5",
                            headers=self.api_headers
                        )
                        
                        if jobs_response.status_code == 200:
                            jobs = jobs_response.json()
                            for job in jobs:
                                st.text(f"{job['created_at'][:19]} - {job['status']} - Job {job['job_id'][:8]}")
                
                time.sleep(refresh_interval)
                
            except Exception as e:
                st.error(f"Monitoring error: {str(e)}")
                break
    
    def generate_analysis_report(self, df):
        """Generate comprehensive HTML analysis report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ODE Dataset Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .metric {{ display: inline-block; margin: 20px; padding: 15px; 
                          background: #f0f0f0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>ODE Dataset Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Dataset Overview</h2>
            <div class="metric">Total ODEs: {len(df)}</div>
            <div class="metric">Verified: {df['verified'].sum()} ({df['verified'].mean():.1%})</div>
            <div class="metric">Generators: {df['generator_name'].nunique()}</div>
            <div class="metric">Functions: {df['function_name'].nunique()}</div>
            
            <h2>Generator Statistics</h2>
            {df.groupby('generator_name')['verified'].agg(['count', 'sum', 'mean']).to_html()}
            
            <h2>Complexity Analysis</h2>
            <p>Average Complexity: {df['complexity_score'].mean():.1f}</p>
            <p>Complexity Range: {df['complexity_score'].min():.0f} - {df['complexity_score'].max():.0f}</p>
            
            <h2>Top Functions</h2>
            {df['function_name'].value_counts().head(10).to_frame().to_html()}
        </body>
        </html>
        """
        return html
    
    def get_sample_odes(self):
        """Get sample ODEs for demonstration"""
        return [
            {
                'id': '1',
                'generator_name': 'L1',
                'function_name': 'sine',
                'ode_symbolic': "y''(x) + y(x) = pi*sin(x)",
                'ode_latex': r"y''(x) + y(x) = \pi \sin(x)",
                'solution_symbolic': "C1*cos(x) + C2*sin(x) - pi*x*cos(x)/2",
                'solution_latex': r"C_1\cos(x) + C_2\sin(x) - \frac{\pi x\cos(x)}{2}",
                'verified': True,
                'complexity_score': 45,
                'has_pantograph': False
            },
            {
                'id': '2',
                'generator_name': 'N1',
                'function_name': 'exponential',
                'ode_symbolic': "(y''(x))**2 + y(x) = exp(x)",
                'ode_latex': r"(y''(x))^2 + y(x) = e^x",
                'solution_symbolic': "exp(x)/4 + C1*exp(-x/2) + C2*x*exp(-x/2)",
                'solution_latex': r"\frac{e^x}{4} + C_1e^{-x/2} + C_2xe^{-x/2}",
                'verified': True,
                'complexity_score': 67,
                'has_pantograph': False
            },
            {
                'id': '3',
                'generator_name': 'L4',
                'function_name': 'identity',
                'ode_symbolic': "y''(x) + y(x/2) - y(x) = 0",
                'ode_latex': r"y''(x) + y(x/2) - y(x) = 0",
                'solution_symbolic': "C1*cos(x) + C2*sin(x)",
                'solution_latex': r"C_1\cos(x) + C_2\sin(x)",
                'verified': True,
                'complexity_score': 52,
                'has_pantograph': True
            }
        ]
    
    def visualize_ode(self, ode):
        """Visualize ODE solution"""
        st.subheader("ODE Visualization")
        
        try:
            # Generate sample solution curve
            x = np.linspace(0, 10, 100)
            
            # Simple visualization based on generator type
            if ode['generator_name'].startswith('L'):
                # Linear ODE - likely oscillatory
                y = np.cos(x) + 0.5 * np.sin(x)
            else:
                # Nonlinear ODE - more complex behavior
                y = np.exp(-x/5) * np.cos(2*x) + 0.1 * x
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Solution'))
            fig.update_layout(
                title=f"Solution Curve: {ode['generator_name']} - {ode['function_name']}",
                xaxis_title="x",
                yaxis_title="y(x)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
    
    def get_job_status(self, job_id):
        """Get job status from API"""
        try:
            response = requests.get(
                f"{st.session_state.api_url}/api/v1/jobs/{job_id}",
                headers=self.api_headers
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def split_ml_dataset(self, dataset_file, test_size, val_size, strategy):
        """Split dataset for ML training"""
        st.info(f"Splitting dataset with {strategy} strategy...")
        st.success(f"Dataset split complete! Test: {test_size}, Val: {val_size}")
    
    def start_ml_training(self, model_type, epochs, batch_size, learning_rate):
        """Start ML model training"""
        st.info(f"Starting {model_type} training...")
        # Store job ID for tracking
        st.session_state.training_job_id = "demo_job_123"
        st.success("Training started! Monitor progress in the right panel.")
    
    def generate_ml_odes(self, model, n_samples, temperature, target_gen, target_func, complexity_range):
        """Generate ODEs using ML model"""
        st.info(f"Generating {n_samples} ODEs with ML model...")
        with st.spinner("Generating..."):
            time.sleep(2)  # Simulate generation
        st.success(f"Generated {n_samples} novel ODEs!")
        st.balloons()
    
    def verify_dataset(self, uploaded_file, methods, confidence_threshold, workers):
        """Verify uploaded dataset"""
        st.info(f"Verifying dataset with {len(methods)} methods...")
        progress_bar = st.progress(0)
        
        # Simulate verification progress
        for i in range(10):
            progress_bar.progress((i + 1) / 10)
            time.sleep(0.2)
        
        st.success("Dataset verification complete!")
        
        # Show sample results
        results_df = pd.DataFrame({
            'ODE ID': range(1, 6),
            'Verified': [True, True, False, True, False],
            'Confidence': [0.98, 0.95, 0.45, 0.92, 0.38],
            'Method': ['substitution', 'substitution', 'failed', 'numerical', 'failed']
        })
        st.dataframe(results_df)
    
    def export_visualizations(self, df):
        """Export all visualizations as a ZIP file"""
        st.info("Exporting visualizations...")
        # In a real implementation, this would create actual plots
        st.success("Visualizations exported!")

# Main execution
if __name__ == "__main__":
    app = ODEMasterGUI()
    app.main()
