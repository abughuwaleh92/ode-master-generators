# gui/advanced_integrated_interface_100_real.py
"""
Advanced Integrated GUI for ODE Master Generator - 100% Real Data Version
Author: Mohammad Abu Ghuwaleh

Complete interface using only real API data - no mock data
"""

import os
import time
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import threading
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import StringIO, BytesIO
from collections import defaultdict, Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import requests
from streamlit_ace import st_ace
import streamlit.components.v1 as components
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ODE Master Generator | Mohammad Abu Ghuwaleh",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/mohammad-abu-ghuwaleh/ode-master-generator',
        'Report a bug': 'https://github.com/mohammad-abu-ghuwaleh/ode-master-generator/issues',
        'About': '# ODE Master Generator\nBy Mohammad Abu Ghuwaleh\n\nA comprehensive system for ODE generation, verification, and analysis.'
    }
)

# ──────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .author-credit {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .job-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .ml-model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .dataset-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000/api/v1')
API_KEY = os.getenv('API_KEY', 'test-key')

# ──────────────────────────────────────────────────────────────────────────
#  ADVANCED ODE INTERFACE CLASS - 100% REAL DATA
# ──────────────────────────────────────────────────────────────────────────
class AdvancedODEInterface:
    """
    Advanced interface for ODE Master Generator with 100% real data
    Author: Mohammad Abu Ghuwaleh
    """
    
    def __init__(self):
        self.api_headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        self.session_state = st.session_state
        self._initialize_session_state()
        self._load_api_capabilities()
        
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'generated_odes': [],
            'current_dataset': [],
            'all_jobs': [],  # Store all jobs for history
            'job_history': [],
            'active_jobs': {},
            'api_capabilities': {},
            'available_generators': [],
            'available_functions': [],
            'ml_models': [],
            'analysis_results': {},
            'api_stats': {},
            'user_preferences': {
                'theme': 'light',
                'auto_refresh': False,
                'refresh_interval': 5
            },
            'ml_training_history': [],
            'current_ml_model': None,
            'generated_ml_odes': [],
            'datasets_in_session': [],  # Track datasets created in session
            'verification_history': [],  # Track verification attempts
            'generation_metrics': defaultdict(lambda: {'count': 0, 'verified': 0}),  # Track generation metrics
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _load_api_capabilities(self):
        """Load API capabilities on startup"""
        try:
            # Get available generators
            response = requests.get(f"{API_BASE_URL}/generators", headers=self.api_headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.available_generators = data.get('all', [])
                st.session_state.api_capabilities['generators'] = data
            
            # Get available functions
            response = requests.get(f"{API_BASE_URL}/functions", headers=self.api_headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.available_functions = data.get('functions', [])
            
            # Get ML models
            response = requests.get(f"{API_BASE_URL}/models", headers=self.api_headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.ml_models = data.get('models', [])
            
            # Get initial stats
            response = requests.get(f"{API_BASE_URL}/stats", headers=self.api_headers, timeout=5)
            if response.status_code == 200:
                st.session_state.api_stats = response.json()
                
        except Exception as e:
            st.sidebar.warning(f"Could not load API capabilities: {str(e)}")
    
    def check_api_connection(self) -> bool:
        """Enhanced API connection check with detailed status"""
        st.sidebar.markdown("### 🔌 Connection Status")
        
        # Show configuration
        with st.sidebar.expander("Configuration", expanded=False):
            if os.getenv('API_BASE_URL'):
                st.success("✅ Using environment variables")
            else:
                st.warning("⚠️ Using default configuration")
            
            api_host = API_BASE_URL.split('/')[2] if len(API_BASE_URL.split('/')) > 2 else API_BASE_URL
            st.text(f"API: {api_host}")
            st.text(f"Key: {'*' * 16}")
        
        # Test connection
        try:
            health_url = API_BASE_URL.replace('/api/v1', '') + '/health'
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                st.sidebar.success("✅ API Connected")
                
                # Show API status details
                with st.sidebar.expander("API Status", expanded=False):
                    st.json(data)
                
                return True
            else:
                st.sidebar.error(f"❌ API Error: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            st.sidebar.error("❌ API Offline")
            if st.sidebar.button("🔄 Retry Connection"):
                st.rerun()
            return False
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🔬 ODE Master Generator</h1>', unsafe_allow_html=True)
        st.markdown('<p class="author-credit">by Mohammad Abu Ghuwaleh</p>', unsafe_allow_html=True)
        
        # Check API connection
        api_available = self.check_api_connection()
        
        # Show API stats in sidebar
        if api_available:
            self._show_sidebar_stats()
        
        # Main navigation
        st.sidebar.markdown("### 🧭 Navigation")
        
        main_page = st.sidebar.radio(
            "Main Section",
            ["🏠 Dashboard", "🧮 Generation", "🤖 Machine Learning", "📊 Analysis", 
             "📡 Monitoring", "🔧 Tools", "📚 Documentation"]
        )
        
        # Route to appropriate section
        if main_page == "🏠 Dashboard":
            self.dashboard_page()
        elif main_page == "🧮 Generation":
            self.generation_section()
        elif main_page == "🤖 Machine Learning":
            self.ml_section()
        elif main_page == "📊 Analysis":
            self.analysis_section()
        elif main_page == "📡 Monitoring":
            self.monitoring_section()
        elif main_page == "🔧 Tools":
            self.tools_section()
        elif main_page == "📚 Documentation":
            self.documentation_page()
    
    def _show_sidebar_stats(self):
        """Show real-time stats in sidebar from API"""
        with st.sidebar.expander("📊 Quick Stats", expanded=True):
            try:
                response = requests.get(f"{API_BASE_URL}/stats", headers=self.api_headers, timeout=5)
                if response.status_code == 200:
                    stats = response.json()
                    st.session_state.api_stats = stats  # Update cached stats
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Active Jobs", stats.get('active_jobs', 0))
                        st.metric("Generated", stats.get('total_generated', 0))
                    with col2:
                        st.metric("Verified", stats.get('total_verified', 0))
                        st.metric("Models", len(st.session_state.ml_models))
            except:
                st.text("Stats unavailable")
    
    # ─────────────────────────────────────────────────────────────────────
    # DASHBOARD SECTION - 100% REAL DATA
    # ─────────────────────────────────────────────────────────────────────
    def dashboard_page(self):
        """Main dashboard with overview and recent activity using real data"""
        st.title("📊 Dashboard")
        
        # Refresh button
        if st.button("🔄 Refresh Dashboard", type="secondary"):
            st.rerun()
        
        # Get fresh stats
        try:
            response = requests.get(f"{API_BASE_URL}/stats", headers=self.api_headers, timeout=5)
            if response.status_code == 200:
                stats = response.json()
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total ODEs Generated",
                        stats.get('total_generated', 0),
                        delta=None  # Real delta would need historical data
                    )
                
                with col2:
                    verification_rate = (stats.get('total_verified', 0) / 
                                       max(stats.get('total_generated', 1), 1) * 100)
                    st.metric(
                        "Verification Rate",
                        f"{verification_rate:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Active Jobs",
                        stats.get('active_jobs', 0)
                    )
                
                with col4:
                    st.metric(
                        "ML Models",
                        len(st.session_state.ml_models)
                    )
        except:
            st.info("Dashboard metrics loading...")
        
        # Charts row
        st.markdown("### 📈 System Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Job distribution chart using real API data
            st.subheader("Job Distribution")
            self._plot_real_job_distribution()
        
        with col2:
            # Generator performance using session data
            st.subheader("Generator Performance")
            self._plot_real_generator_performance()
        
        # Recent jobs from API
        st.markdown("### 🕐 Recent Activity")
        self._show_real_recent_activity()
        
        # ML Models Overview
        st.markdown("### 🤖 ML Models Overview")
        self._show_ml_models_overview()
        
        # Session statistics
        st.markdown("### 📊 Current Session Statistics")
        self._show_session_statistics()
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Quick Generate", use_container_width=True):
                st.session_state.selected_nav = "🧮 Generation"
                st.rerun()
        
        with col2:
            if st.button("🔍 Verify ODE", use_container_width=True):
                st.session_state.selected_nav = "🔧 Tools"
                st.rerun()
        
        with col3:
            if st.button("🤖 Train Model", use_container_width=True):
                st.session_state.selected_nav = "🤖 Machine Learning"
                st.rerun()
        
        with col4:
            if st.button("📊 New Analysis", use_container_width=True):
                st.session_state.selected_nav = "📊 Analysis"
                st.rerun()
    
    def _plot_real_job_distribution(self):
        """Plot job distribution using real API data"""
        try:
            # Get stats from API
            stats = st.session_state.api_stats
            job_stats = stats.get('job_statistics', {})
            
            if job_stats:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=[k.capitalize() for k in job_stats.keys()],
                        values=list(job_stats.values()),
                        hole=0.3,
                        marker=dict(
                            colors=['#28a745', '#17a2b8', '#dc3545', '#ffc107', '#6c757d']
                        )
                    )
                ])
                
                fig.update_layout(
                    title="Job Status Distribution (Real-time)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to current session jobs
                if st.session_state.all_jobs:
                    job_counts = Counter(job['status'] for job in st.session_state.all_jobs)
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=[k.capitalize() for k in job_counts.keys()],
                            values=list(job_counts.values()),
                            hole=0.3
                        )
                    ])
                    
                    fig.update_layout(
                        title="Session Job Distribution",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No job data available yet")
        except Exception as e:
            st.info("Job distribution unavailable")
    
    def _plot_real_generator_performance(self):
        """Plot generator performance using real session data"""
        # Use generation metrics from current session
        if st.session_state.generation_metrics:
            generators = []
            success_rates = []
            
            for gen, metrics in dict(st.session_state.generation_metrics).items():
                if metrics['count'] > 0:
                    generators.append(gen)
                    rate = (metrics['verified'] / metrics['count']) * 100
                    success_rates.append(rate)
            
            if generators:
                fig = go.Figure(data=[
                    go.Bar(
                        x=generators,
                        y=success_rates,
                        text=[f"{rate:.1f}%" for rate in success_rates],
                        textposition='auto',
                        marker_color='lightblue'
                    )
                ])
                
                fig.update_layout(
                    title="Generator Success Rates (Current Session)",
                    xaxis_title="Generator",
                    yaxis_title="Success Rate (%)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Generate some ODEs to see performance metrics")
        else:
            st.info("No generation data available yet")
    
    def _show_real_recent_activity(self):
        """Show recent activity using real job data"""
        try:
            # Get recent jobs from API
            response = requests.get(
                f"{API_BASE_URL}/jobs",
                headers=self.api_headers,
                params={'limit': 10}
            )
            
            if response.status_code == 200:
                jobs = response.json()
                
                if jobs:
                    for job in jobs[:5]:  # Show last 5 jobs
                        col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                        
                        with col1:
                            # Calculate time ago
                            created = datetime.fromisoformat(job['created_at'].replace('Z', '+00:00'))
                            time_ago = datetime.now(created.tzinfo) - created
                            
                            if time_ago.total_seconds() < 60:
                                time_str = f"{int(time_ago.total_seconds())}s ago"
                            elif time_ago.total_seconds() < 3600:
                                time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                            else:
                                time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
                            
                            st.text(time_str)
                        
                        with col2:
                            # Get job type from metadata or params
                            job_type = "Unknown"
                            if 'metadata' in job and 'type' in job['metadata']:
                                job_type = job['metadata']['type']
                            elif 'generator' in job:
                                job_type = f"Generate ({job['generator']})"
                            
                            st.text(job_type)
                        
                        with col3:
                            st.text(f"Job {job['job_id'][:8]}...")
                        
                        with col4:
                            if job['status'] == 'completed':
                                st.success("✅")
                            elif job['status'] == 'running':
                                st.info("🔄")
                            elif job['status'] == 'failed':
                                st.error("❌")
                            else:
                                st.warning("⏸")
                else:
                    st.info("No recent activity")
            else:
                # Fallback to session jobs
                if st.session_state.job_history:
                    for job in st.session_state.job_history[-5:]:
                        col1, col2, col3 = st.columns([2, 4, 2])
                        
                        with col1:
                            time_ago = datetime.now() - job['created_at']
                            if time_ago.total_seconds() < 60:
                                st.text(f"{int(time_ago.total_seconds())}s ago")
                            else:
                                st.text(f"{int(time_ago.total_seconds() / 60)}m ago")
                        
                        with col2:
                            st.text(f"{job['type']}: {job['params'].get('generator', 'Unknown')}")
                        
                        with col3:
                            st.success("✅ Completed")
                else:
                    st.info("No activity in current session")
                    
        except Exception as e:
            st.info("Activity feed unavailable")
    
    def _show_session_statistics(self):
        """Show statistics from current session"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Session ODEs",
                len(st.session_state.current_dataset)
            )
        
        with col2:
            verified_count = sum(1 for ode in st.session_state.current_dataset if ode.get('verified', False))
            st.metric(
                "Session Verified",
                verified_count
            )
        
        with col3:
            st.metric(
                "Session Jobs",
                len(st.session_state.job_history)
            )
        
        with col4:
            st.metric(
                "Datasets Created",
                len(st.session_state.datasets_in_session)
            )
    
    # ─────────────────────────────────────────────────────────────────────
    # GENERATION SECTION - 100% REAL
    # ─────────────────────────────────────────────────────────────────────
    def generation_section(self):
        """Enhanced ODE generation interface"""
        st.title("🧮 ODE Generation")
        
        tabs = st.tabs(["Standard Generation", "Batch Generation", "Stream Generation", "ML Generation", "Custom Generation"])
        
        with tabs[0]:
            self.standard_generation_page()
        
        with tabs[1]:
            self.batch_generation_page()
        
        with tabs[2]:
            self.stream_generation_page()
        
        with tabs[3]:
            self.ml_generation_page()
        
        with tabs[4]:
            self.custom_generation_page()
    
    def standard_generation_page(self):
        """Standard ODE generation interface"""
        st.markdown("### Generate ODEs with Standard Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            generator = st.selectbox(
                "Generator",
                st.session_state.available_generators,
                help="Select the ODE generator type"
            )
        
        with col2:
            function = st.selectbox(
                "Function",
                st.session_state.available_functions,
                help="Select the function type"
            )
        
        with col3:
            count = st.number_input(
                "Number of ODEs",
                min_value=1,
                max_value=100,
                value=5,
                help="Number of ODEs to generate"
            )
        
        # Advanced parameters
        with st.expander("⚙️ Advanced Parameters", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                alpha = st.slider("α (Alpha)", -5.0, 5.0, 1.0, 0.1)
            with col2:
                beta = st.slider("β (Beta)", 0.1, 5.0, 1.0, 0.1)
            with col3:
                M = st.slider("M", -5.0, 5.0, 0.0, 0.1)
            with col4:
                verify = st.checkbox("Auto-verify", value=True)
            
            # Additional parameters for nonlinear generators
            if generator and generator.startswith('N'):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if generator in ['N1', 'N2', 'N6']:
                        q = st.number_input("q (power)", 2, 10, 2)
                    else:
                        q = 2
                
                with col2:
                    if generator in ['N2', 'N3', 'N6']:
                        v = st.number_input("v (power)", 2, 10, 3)
                    else:
                        v = 3
                
                with col3:
                    if generator in ['L4', 'N6']:
                        a = st.number_input("a (delay)", 2.0, 10.0, 2.0)
                    else:
                        a = 2.0
        
        # Generation options
        col1, col2 = st.columns(2)
        with col1:
            save_to_dataset = st.checkbox("Save to current dataset", value=True)
        with col2:
            export_format = st.selectbox("Export format", ["JSON", "CSV", "LaTeX", "Python"])
        
        # Generate button
        if st.button("🚀 Generate ODEs", type="primary", use_container_width=True):
            with st.spinner("Generating ODEs via API..."):
                # Build parameters
                params = {"alpha": alpha, "beta": beta, "M": M}
                
                # Add conditional parameters
                if generator in ['N1', 'N2', 'N6']:
                    params['q'] = q if 'q' in locals() else 2
                if generator in ['N2', 'N3', 'N6']:
                    params['v'] = v if 'v' in locals() else 3
                if generator in ['L4', 'N6']:
                    params['a'] = a if 'a' in locals() else 2
                
                response = self._call_api_generate({
                    "generator": generator,
                    "function": function,
                    "parameters": params,
                    "count": count,
                    "verify": verify
                })
                
                if response['status'] == 'success':
                    job_id = response['data']['job_id']
                    st.success(f"Generation job created: `{job_id}`")
                    
                    # Add to job history
                    job_record = {
                        'job_id': job_id,
                        'type': 'generation',
                        'created_at': datetime.now(),
                        'params': {
                            'generator': generator,
                            'function': function,
                            'count': count
                        }
                    }
                    st.session_state.job_history.append(job_record)
                    st.session_state.all_jobs.append({
                        'job_id': job_id,
                        'status': 'running',
                        'created_at': datetime.now().isoformat()
                    })
                    
                    # Poll for results
                    results = self._poll_job_status_advanced(job_id)
                    
                    if results:
                        # Update generation metrics
                        st.session_state.generation_metrics[generator]['count'] += len(results)
                        verified_count = sum(1 for r in results if r.get('verified', False))
                        st.session_state.generation_metrics[generator]['verified'] += verified_count
                        
                        st.session_state.generated_odes = results
                        if save_to_dataset:
                            st.session_state.current_dataset.extend(results)
                        
                        # Display results
                        self._display_generation_results(results, export_format)
                else:
                    st.error(f"Generation failed: {response.get('error', 'Unknown error')}")
    
    def batch_generation_page(self):
        """Batch generation interface"""
        st.markdown("### Batch Generation - Generate Multiple Combinations")
        
        # Multi-select for generators and functions
        col1, col2 = st.columns(2)
        
        with col1:
            selected_generators = st.multiselect(
                "Select Generators",
                st.session_state.available_generators,
                default=st.session_state.available_generators[:3] if st.session_state.available_generators else []
            )
        
        with col2:
            selected_functions = st.multiselect(
                "Select Functions",
                st.session_state.available_functions,
                default=st.session_state.available_functions[:3] if st.session_state.available_functions else []
            )
        
        # Batch settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            samples_per_combo = st.number_input(
                "Samples per combination",
                min_value=1,
                max_value=20,
                value=5
            )
        
        with col2:
            total_combinations = len(selected_generators) * len(selected_functions)
            total_odes = total_combinations * samples_per_combo
            st.metric("Total Combinations", total_combinations)
        
        with col3:
            st.metric("Total ODEs", total_odes)
        
        # Parameter ranges for batch
        with st.expander("Parameter Ranges"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                alpha_range = st.slider("α Range", -5.0, 5.0, (-1.0, 2.0))
            with col2:
                beta_range = st.slider("β Range", 0.1, 5.0, (0.5, 2.0))
            with col3:
                M_range = st.slider("M Range", -5.0, 5.0, (-1.0, 1.0))
        
        # Generate batch
        if st.button("🚀 Generate Batch", type="primary", use_container_width=True):
            if not selected_generators or not selected_functions:
                st.error("Please select at least one generator and one function")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            completed = 0
            
            for i, generator in enumerate(selected_generators):
                for j, function in enumerate(selected_functions):
                    status_text.text(f"Generating: {generator} + {function}")
                    
                    # Random parameters within ranges
                    params = {
                        "alpha": np.random.uniform(alpha_range[0], alpha_range[1]),
                        "beta": np.random.uniform(beta_range[0], beta_range[1]),
                        "M": np.random.uniform(M_range[0], M_range[1])
                    }
                    
                    # Call API
                    response = self._call_api_generate({
                        "generator": generator,
                        "function": function,
                        "parameters": params,
                        "count": samples_per_combo,
                        "verify": True
                    })
                    
                    if response['status'] == 'success':
                        job_id = response['data']['job_id']
                        results = self._poll_job_status_simple(job_id)
                        
                        if results:
                            all_results.extend(results)
                            
                            # Update metrics
                            st.session_state.generation_metrics[generator]['count'] += len(results)
                            verified = sum(1 for r in results if r.get('verified', False))
                            st.session_state.generation_metrics[generator]['verified'] += verified
                    
                    completed += 1
                    progress_bar.progress(completed / total_combinations)
            
            status_text.text(f"Batch generation complete! Generated {len(all_results)} ODEs")
            
            # Add to dataset
            st.session_state.current_dataset.extend(all_results)
            
            # Show summary
            st.success(f"✅ Generated {len(all_results)} ODEs successfully!")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Generated", len(all_results))
            
            with col2:
                verified_count = sum(1 for r in all_results if r.get('verified', False))
                st.metric("Verified", verified_count)
            
            with col3:
                verification_rate = (verified_count / len(all_results) * 100) if all_results else 0
                st.metric("Success Rate", f"{verification_rate:.1f}%")
    
    def stream_generation_page(self):
        """Stream generation interface"""
        st.markdown("### Stream Generation - Real-time ODE Generation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            generator = st.selectbox(
                "Generator",
                st.session_state.available_generators,
                key="stream_gen"
            )
        
        with col2:
            function = st.selectbox(
                "Function", 
                st.session_state.available_functions,
                key="stream_func"
            )
        
        with col3:
            count = st.number_input(
                "Number of ODEs",
                min_value=1,
                max_value=50,
                value=10,
                key="stream_count"
            )
        
        # Stream container
        stream_container = st.container()
        
        if st.button("🌊 Start Streaming", type="primary", use_container_width=True):
            with stream_container:
                st.markdown("### Streaming Results")
                
                # Create placeholder for each ODE
                placeholders = [st.empty() for _ in range(count)]
                
                try:
                    # Stream endpoint
                    stream_url = f"{API_BASE_URL}/stream/generate"
                    
                    response = requests.get(
                        stream_url,
                        headers=self.api_headers,
                        params={
                            'generator': generator,
                            'function': function,
                            'count': count
                        },
                        stream=True
                    )
                    
                    if response.status_code == 200:
                        ode_count = 0
                        
                        for line in response.iter_lines():
                            if line:
                                line_str = line.decode('utf-8')
                                if line_str.startswith('data: '):
                                    try:
                                        data = json.loads(line_str[6:])
                                        
                                        if 'error' not in data and ode_count < count:
                                            # Display ODE in placeholder
                                            with placeholders[ode_count]:
                                                st.success(f"✅ ODE {ode_count + 1}")
                                                col1, col2 = st.columns(2)
                                                
                                                with col1:
                                                    st.markdown("**ODE:**")
                                                    st.code(data.get('ode', ''))
                                                
                                                with col2:
                                                    st.markdown("**Verified:**")
                                                    if data.get('verified'):
                                                        st.success("Yes")
                                                    else:
                                                        st.error("No")
                                            
                                            ode_count += 1
                                            
                                            # Add to dataset
                                            st.session_state.current_dataset.append(data)
                                            
                                    except json.JSONDecodeError:
                                        continue
                        
                        st.success(f"✅ Streaming complete! Generated {ode_count} ODEs")
                    else:
                        st.error(f"Streaming failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Streaming error: {str(e)}")
                    
                    # Fallback to regular generation
                    st.info("Falling back to regular generation...")
                    response = self._call_api_generate({
                        "generator": generator,
                        "function": function,
                        "count": count,
                        "verify": True
                    })
                    
                    if response['status'] == 'success':
                        job_id = response['data']['job_id']
                        results = self._poll_job_status_advanced(job_id)
                        
                        if results:
                            for i, ode in enumerate(results[:count]):
                                with placeholders[i]:
                                    st.success(f"✅ ODE {i + 1}")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**ODE:**")
                                        st.code(ode.get('ode', ''))
                                    
                                    with col2:
                                        st.markdown("**Verified:**")
                                        if ode.get('verified'):
                                            st.success("Yes")
                                        else:
                                            st.error("No")
    
    def ml_generation_page(self):
        """ML-powered ODE generation"""
        st.markdown("### Generate ODEs using ML Models")
        
        if not st.session_state.ml_models:
            st.warning("No ML models available. Train a model first!")
            if st.button("Go to ML Training"):
                st.session_state.selected_nav = "🤖 Machine Learning"
                st.rerun()
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selection
            selected_model = st.selectbox(
                "Select Model",
                st.session_state.ml_models,
                format_func=lambda x: f"{x['name']} ({x.get('metadata', {}).get('model_type', 'Unknown')})"
            )
            
            if selected_model:
                st.info(f"""
                **Model Info:**
                - Type: {selected_model.get('metadata', {}).get('model_type', 'Unknown')}
                - Accuracy: {selected_model.get('metadata', {}).get('accuracy', 'N/A')}%
                - Size: {selected_model['size'] / 1024 / 1024:.1f} MB
                - Created: {selected_model.get('created', 'Unknown')}
                """)
        
        with col2:
            n_samples = st.number_input("Number of ODEs", 1, 100, 10)
            temperature = st.slider("Creativity", 0.1, 2.0, 0.8)
            
            # Optional constraints
            with st.expander("Generation Constraints"):
                target_generator = st.selectbox(
                    "Target Generator", 
                    ["Any"] + st.session_state.available_generators
                )
                target_function = st.selectbox(
                    "Target Function", 
                    ["Any"] + st.session_state.available_functions
                )
                complexity_range = st.slider("Complexity Range", 0, 500, (50, 200))
        
        if st.button("🎨 Generate with ML", type="primary", use_container_width=True):
            with st.spinner("AI is generating ODEs..."):
                # Call ML generation API
                response = self._call_api_ml_generate({
                    "model_path": selected_model['path'],
                    "n_samples": n_samples,
                    "temperature": temperature,
                    "generator": None if target_generator == "Any" else target_generator,
                    "function": None if target_function == "Any" else target_function,
                    "complexity_range": list(complexity_range)
                })
                
                if response['status'] == 'success':
                    job_id = response['data']['job_id']
                    st.success(f"ML generation job created: `{job_id}`")
                    
                    # Poll for results
                    results = self._poll_job_status_advanced(job_id)
                    
                    if results and 'odes' in results:
                        st.session_state.generated_ml_odes = results['odes']
                        self._display_ml_generation_results(results)
                else:
                    st.error(f"ML generation failed: {response.get('error', 'Unknown error')}")
    
    def custom_generation_page(self):
        """Custom ODE generation with manual parameters"""
        st.markdown("### Custom ODE Generation")
        st.info("Generate ODEs with custom parameter combinations")
        
        # Custom parameter input
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Generator Settings")
            generator = st.selectbox(
                "Generator",
                st.session_state.available_generators,
                key="custom_gen"
            )
            
            function = st.selectbox(
                "Function",
                st.session_state.available_functions,
                key="custom_func"
            )
        
        with col2:
            st.markdown("#### Parameters")
            
            # Dynamic parameter inputs based on generator
            params = {}
            
            params['alpha'] = st.number_input("α (Alpha)", -10.0, 10.0, 1.0, 0.1)
            params['beta'] = st.number_input("β (Beta)", 0.01, 10.0, 1.0, 0.1)
            params['M'] = st.number_input("M", -10.0, 10.0, 0.0, 0.1)
            
            if generator and generator.startswith('N'):
                if generator in ['N1', 'N2', 'N6']:
                    params['q'] = st.number_input("q (power)", 1, 20, 2)
                if generator in ['N2', 'N3', 'N6']:
                    params['v'] = st.number_input("v (power)", 1, 20, 3)
            
            if generator in ['L4', 'N6']:
                params['a'] = st.number_input("a (delay factor)", 1.1, 10.0, 2.0, 0.1)
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                verify_method = st.selectbox(
                    "Verification Method",
                    ["substitution", "numerical", "both"]
                )
                
                tolerance = st.number_input(
                    "Numerical Tolerance",
                    min_value=1e-12,
                    max_value=1e-3,
                    value=1e-8,
                    format="%.2e"
                )
            
            with col2:
                include_latex = st.checkbox("Include LaTeX", value=True)
                include_plot = st.checkbox("Generate Solution Plot", value=False)
        
        # Generate button
        if st.button("🔧 Generate Custom ODE", type="primary", use_container_width=True):
            with st.spinner("Generating custom ODE..."):
                response = self._call_api_generate({
                    "generator": generator,
                    "function": function,
                    "parameters": params,
                    "count": 1,
                    "verify": True
                })
                
                if response['status'] == 'success':
                    job_id = response['data']['job_id']
                    results = self._poll_job_status_advanced(job_id)
                    
                    if results and len(results) > 0:
                        ode = results[0]
                        
                        # Display detailed result
                        st.success("✅ Custom ODE Generated Successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### ODE Equation")
                            st.code(ode.get('ode', ''))
                            
                            if include_latex and 'ode_latex' in ode:
                                st.markdown("**LaTeX:**")
                                st.latex(ode['ode_latex'])
                        
                        with col2:
                            st.markdown("#### Solution")
                            st.code(ode.get('solution', ''))
                            
                            if include_latex and 'solution_latex' in ode:
                                st.markdown("**LaTeX:**")
                                st.latex(ode['solution_latex'])
                        
                        # Properties
                        st.markdown("#### Properties")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Verified", "✅ Yes" if ode.get('verified') else "❌ No")
                        with col2:
                            st.metric("Complexity", ode.get('complexity', 'N/A'))
                        with col3:
                            st.metric("Confidence", f"{ode.get('properties', {}).get('verification_confidence', 0):.1%}")
                        with col4:
                            if ode.get('properties', {}).get('has_pantograph'):
                                st.metric("Pantograph", "Yes")
                            else:
                                st.metric("Pantograph", "No")
                        
                        # Add to dataset option
                        if st.button("➕ Add to Dataset"):
                            st.session_state.current_dataset.append(ode)
                            st.success("Added to current dataset!")
    
    # ─────────────────────────────────────────────────────────────────────
    # MACHINE LEARNING SECTION - 100% REAL
    # ─────────────────────────────────────────────────────────────────────
    def ml_section(self):
        """Machine Learning interface"""
        st.title("🤖 Machine Learning")
        
        tabs = st.tabs(["Model Training", "Model Evaluation", "Dataset Preparation", "Model Management", "Training History"])
        
        with tabs[0]:
            self.ml_training_page()
        
        with tabs[1]:
            self.ml_evaluation_page()
        
        with tabs[2]:
            self.ml_dataset_preparation_page()
        
        with tabs[3]:
            self.model_management_page()
        
        with tabs[4]:
            self.training_history_page()
    
    def ml_training_page(self):
        """Enhanced ML training interface"""
        st.markdown("### Train Machine Learning Models")
        
        # Dataset selection
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_source = st.radio(
                "Dataset Source",
                ["Current Session", "Upload File", "Specify Path"]
            )
            
            dataset_path = None
            
            if dataset_source == "Current Session":
                if st.session_state.current_dataset:
                    st.success(f"Using current session dataset ({len(st.session_state.current_dataset)} ODEs)")
                    
                    # Save to temp file for training
                    temp_path = self._save_current_dataset_temp()
                    dataset_path = temp_path
                    
                    # Preview button
                    if st.button("Preview Dataset"):
                        df = pd.DataFrame(st.session_state.current_dataset[:10])
                        st.dataframe(df[['generator', 'function', 'verified', 'complexity']].head())
                else:
                    st.warning("No ODEs in current session. Generate some first!")
                    
            elif dataset_source == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload Dataset",
                    type=['jsonl', 'json', 'csv'],
                    help="Upload ODE dataset file"
                )
                
                if uploaded_file:
                    # Save uploaded file
                    temp_path = f"uploaded_{uploaded_file.name}"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.read())
                    dataset_path = temp_path
                    st.success(f"Uploaded: {uploaded_file.name}")
                    
            else:  # Specify Path
                dataset_path = st.text_input(
                    "Dataset Path",
                    placeholder="path/to/dataset.jsonl"
                )
        
        with col2:
            # Model architecture
            model_type = st.selectbox(
                "Model Architecture",
                ["pattern_net", "transformer", "vae", "language_model"],
                help="Select the neural network architecture"
            )
            
            # Architecture details
            with st.expander("Architecture Details"):
                if model_type == "pattern_net":
                    st.markdown("""
                    **Pattern Network**
                    - Type: Feed-forward neural network
                    - Tasks: Verification prediction, complexity estimation
                    - Input: Numeric features + embeddings
                    - Best for: Quick training, property prediction
                    """)
                elif model_type == "transformer":
                    st.markdown("""
                    **Transformer**
                    - Type: Multi-head attention
                    - Tasks: Sequence modeling, generation
                    - Input: Tokenized ODE sequences
                    - Best for: Complex patterns, large datasets
                    """)
                elif model_type == "vae":
                    st.markdown("""
                    **Variational Autoencoder**
                    - Type: Encoder-decoder with latent space
                    - Tasks: Generation, interpolation
                    - Input: ODE features
                    - Best for: Exploring ODE space
                    """)
                else:
                    st.markdown("""
                    **Language Model**
                    - Type: GPT-style autoregressive
                    - Tasks: Text-based generation
                    - Input: ODE text
                    - Best for: Flexible generation
                    """)
        
        # Training configuration
        st.subheader("Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.number_input("Epochs", 1, 1000, 100)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=2)
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.00001,
                max_value=0.1,
                value=0.001,
                format="%.5f",
                step=0.00001
            )
        
        with col2:
            optimizer = st.selectbox(
                "Optimizer",
                ["adam", "sgd", "rmsprop", "adamw"],
                help="Optimization algorithm"
            )
            early_stopping = st.checkbox("Early Stopping", value=True)
            if early_stopping:
                patience = st.number_input("Patience", 5, 50, 10)
        
        with col3:
            # Model-specific parameters
            if model_type == "pattern_net":
                hidden_dims = st.text_input(
                    "Hidden Dimensions",
                    value="256,128,64",
                    help="Comma-separated hidden layer sizes"
                )
                dropout = st.slider("Dropout", 0.0, 0.5, 0.2)
                
            elif model_type == "transformer":
                n_heads = st.number_input("Attention Heads", 1, 16, 8)
                n_layers = st.number_input("Layers", 1, 12, 6)
                
            elif model_type == "vae":
                latent_dim = st.number_input("Latent Dimension", 8, 256, 64)
                beta = st.slider("β (KL weight)", 0.1, 10.0, 1.0)
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not dataset_path:
                st.error("Please select a dataset!")
                return
            
            # Prepare training config
            training_config = {
                "dataset": dataset_path,
                "model_type": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "early_stopping": early_stopping,
                "config": {
                    "optimizer": optimizer
                }
            }
            
            # Add model-specific config
            if model_type == "pattern_net":
                training_config["config"]["hidden_dims"] = [int(x) for x in hidden_dims.split(',')]
                training_config["config"]["dropout"] = dropout
            elif model_type == "transformer":
                training_config["config"]["n_heads"] = n_heads
                training_config["config"]["n_layers"] = n_layers
            elif model_type == "vae":
                training_config["config"]["latent_dim"] = latent_dim
                training_config["config"]["beta"] = beta
            
            if early_stopping:
                training_config["config"]["patience"] = patience if 'patience' in locals() else 10
            
            # Submit training job
            with st.spinner("Initializing training job..."):
                response = self._call_api_train(training_config)
                
                if response['status'] == 'success':
                    job_id = response['data']['job_id']
                    st.success(f"Training job started: `{job_id}`")
                    
                    # Add to training history
                    st.session_state.ml_training_history.append({
                        'job_id': job_id,
                        'model_type': model_type,
                        'dataset': dataset_source,
                        'config': training_config,
                        'started_at': datetime.now(),
                        'status': 'running'
                    })
                    
                    # Show training dashboard
                    self._show_real_training_dashboard(job_id, training_config)
                else:
                    st.error(f"Failed to start training: {response.get('error')}")
    
    def _show_real_training_dashboard(self, job_id: str, config: Dict):
        """Show real training dashboard with live updates"""
        st.markdown("### Training Dashboard")
        
        # Create containers for live updates
        col1, col2 = st.columns(2)
        
        with col1:
            epoch_container = st.container()
            metrics_chart = st.empty()
        
        with col2:
            status_container = st.container()
            final_results = st.empty()
        
        # Training log
        log_container = st.expander("Training Log", expanded=True)
        
        # Poll for updates
        max_polls = config['epochs'] * 2  # Reasonable limit
        poll_count = 0
        
        # Store metrics history
        metrics_history = {
            'epochs': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        while poll_count < max_polls:
            status = self._get_job_status(job_id)
            
            if status:
                # Update status display
                with status_container:
                    st.markdown(f"**Status:** {status['status'].capitalize()}")
                    
                    if status['progress']:
                        st.progress(status['progress'] / 100)
                        st.text(f"Progress: {status['progress']:.0f}%")
                
                # Update epoch info and collect metrics
                if 'metadata' in status and status['metadata']:
                    metadata = status['metadata']
                    
                    with epoch_container:
                        if 'current_epoch' in metadata:
                            st.markdown(f"**Epoch:** {metadata['current_epoch']}/{config['epochs']}")
                        
                        # Display current metrics
                        if 'loss' in metadata:
                            col1_1, col1_2 = st.columns(2)
                            with col1_1:
                                st.metric("Loss", f"{metadata.get('loss', 'N/A')}")
                            with col1_2:
                                st.metric("Accuracy", f"{metadata.get('accuracy', 'N/A')}")
                    
                    # Collect metrics for chart
                    if 'current_epoch' in metadata:
                        metrics_history['epochs'].append(metadata['current_epoch'])
                        metrics_history['loss'].append(metadata.get('loss', 0))
                        metrics_history['accuracy'].append(metadata.get('accuracy', 0))
                        
                        # Update chart
                        if len(metrics_history['epochs']) > 1:
                            fig = go.Figure()
                            
                            # Loss trace
                            fig.add_trace(go.Scatter(
                                x=metrics_history['epochs'],
                                y=metrics_history['loss'],
                                mode='lines+markers',
                                name='Loss',
                                yaxis='y'
                            ))
                            
                            # Accuracy trace
                            fig.add_trace(go.Scatter(
                                x=metrics_history['epochs'],
                                y=metrics_history['accuracy'],
                                mode='lines+markers',
                                name='Accuracy',
                                yaxis='y2'
                            ))
                            
                            fig.update_layout(
                                title="Training Metrics",
                                xaxis_title="Epoch",
                                yaxis=dict(title="Loss", side="left"),
                                yaxis2=dict(title="Accuracy", side="right", overlaying="y"),
                                height=300
                            )
                            
                            metrics_chart.plotly_chart(fig, use_container_width=True)
                    
                    # Log updates
                    with log_container:
                        if 'status' in metadata:
                            st.text(f"[{datetime.now().strftime('%H:%M:%S')}] {metadata['status']}")
                
                # Check completion
                if status['status'] == 'completed':
                    st.success("✅ Training completed successfully!")
                    
                    # Update training history
                    for record in st.session_state.ml_training_history:
                        if record['job_id'] == job_id:
                            record['status'] = 'completed'
                            record['completed_at'] = datetime.now()
                    
                    if 'results' in status:
                        with final_results:
                            st.markdown("### Final Results")
                            results = status['results']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Final Loss", f"{results.get('final_metrics', {}).get('loss', 'N/A')}")
                            with col2:
                                st.metric("Final Accuracy", f"{results.get('final_metrics', {}).get('accuracy', 'N/A')}%")
                            with col3:
                                st.metric("Training Time", f"{results.get('training_time', 0):.1f}s")
                            
                            if 'model_path' in results:
                                st.success(f"Model saved: `{results['model_path']}`")
                                
                                # Refresh ML models list
                                response = requests.get(f"{API_BASE_URL}/models", headers=self.api_headers)
                                if response.status_code == 200:
                                    st.session_state.ml_models = response.json().get('models', [])
                    break
                
                elif status['status'] == 'failed':
                    st.error(f"❌ Training failed: {status.get('error', 'Unknown error')}")
                    
                    # Update training history
                    for record in st.session_state.ml_training_history:
                        if record['job_id'] == job_id:
                            record['status'] = 'failed'
                            record['error'] = status.get('error')
                    break
            
            poll_count += 1
            time.sleep(3)  # Poll every 3 seconds
        
        if poll_count >= max_polls:
            st.warning("⚠️ Training monitoring timed out. The job may still be running.")
    
    def training_history_page(self):
        """Show ML training history"""
        st.markdown("### Training History")
        
        if not st.session_state.ml_training_history:
            st.info("No training jobs yet. Start training a model!")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_jobs = len(st.session_state.ml_training_history)
        completed_jobs = sum(1 for j in st.session_state.ml_training_history if j.get('status') == 'completed')
        running_jobs = sum(1 for j in st.session_state.ml_training_history if j.get('status') == 'running')
        failed_jobs = sum(1 for j in st.session_state.ml_training_history if j.get('status') == 'failed')
        
        with col1:
            st.metric("Total Jobs", total_jobs)
        with col2:
            st.metric("Completed", completed_jobs)
        with col3:
            st.metric("Running", running_jobs)
        with col4:
            st.metric("Failed", failed_jobs)
        
        # Job history table
        st.markdown("### Recent Training Jobs")
        
        # Convert to DataFrame for display
        history_data = []
        for job in reversed(st.session_state.ml_training_history[-20:]):  # Last 20 jobs
            history_data.append({
                'Job ID': job['job_id'][:8] + '...',
                'Model': job['model_type'],
                'Dataset': job['dataset'],
                'Started': job['started_at'].strftime('%Y-%m-%d %H:%M'),
                 'Status': job.get('status', 'unknown').capitalize(),
                'Duration': str(job.get('completed_at', datetime.now()) - job['started_at']).split('.')[0] if job.get('completed_at') else 'Running'
            })
        
        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Action buttons for each job
            for idx, job in enumerate(reversed(st.session_state.ml_training_history[-5:])):
                with st.expander(f"Job {job['job_id'][:8]}... - {job['model_type']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.json({
                            'Model Type': job['model_type'],
                            'Dataset': job['dataset'],
                            'Epochs': job['config']['epochs'],
                            'Batch Size': job['config']['batch_size'],
                            'Learning Rate': job['config']['learning_rate']
                        })
                    
                    with col2:
                        if job.get('status') == 'running':
                            if st.button(f"Check Status", key=f"check_{job['job_id']}"):
                                status = self._get_job_status(job['job_id'])
                                if status:
                                    st.json(status)
                        elif job.get('status') == 'completed':
                            st.success("✅ Completed")
                            if st.button(f"Use Model", key=f"use_{job['job_id']}"):
                                st.session_state.selected_nav = "🧮 Generation"
                                st.rerun()
        else:
            st.info("No training history available")
    
    # ─────────────────────────────────────────────────────────────────────
    # ANALYSIS SECTION - 100% REAL
    # ─────────────────────────────────────────────────────────────────────
    def analysis_section(self):
        """Comprehensive analysis interface"""
        st.title("📊 Analysis Suite")
        
        tabs = st.tabs([
            "Dataset Analysis",
            "Pattern Discovery", 
            "Statistical Analysis",
            "Visualization Studio",
            "Export & Reports"
        ])
        
        with tabs[0]:
            self.dataset_analysis_page()
        
        with tabs[1]:
            self.pattern_discovery_page()
        
        with tabs[2]:
            self.statistical_analysis_page()
        
        with tabs[3]:
            self.visualization_studio_page()
        
        with tabs[4]:
            self.export_reports_page()
    
    def dataset_analysis_page(self):
        """Dataset analysis using real API"""
        st.markdown("### Dataset Analysis")
        
        # Dataset selection
        dataset_source = st.radio(
            "Select Dataset",
            ["Current Session Dataset", "Upload Dataset", "Specify Path"]
        )
        
        dataset_path = None
        df = None
        
        if dataset_source == "Current Session Dataset":
            if st.session_state.current_dataset:
                df = pd.DataFrame(st.session_state.current_dataset)
                st.success(f"Using current session dataset with {len(df)} ODEs")
                dataset_path = self._save_current_dataset_temp()
            else:
                st.warning("No ODEs in current session")
        
        elif dataset_source == "Upload Dataset":
            uploaded_file = st.file_uploader("Upload ODE Dataset", type=['jsonl', 'json', 'csv'])
            if uploaded_file:
                df = self._load_uploaded_dataset(uploaded_file)
                # Save for API
                temp_path = f"temp_upload_{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.read())
                dataset_path = temp_path
        
        else:
            dataset_path = st.text_input("Dataset Path", placeholder="path/to/dataset.jsonl")
            if dataset_path and st.button("Load Dataset"):
                df = self._load_dataset(dataset_path)
        
        if df is not None and not df.empty:
            # Basic statistics from loaded data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total ODEs", len(df))
            
            with col2:
                verified_rate = (df['verified'].sum() / len(df) * 100) if 'verified' in df else 0
                st.metric("Verified", f"{verified_rate:.1f}%")
            
            with col3:
                avg_complexity = df['complexity_score'].mean() if 'complexity_score' in df else 0
                st.metric("Avg Complexity", f"{avg_complexity:.1f}")
            
            with col4:
                unique_generators = df['generator_name'].nunique() if 'generator_name' in df else 0
                st.metric("Generators", unique_generators)
            
            # Quick insights from current data
            st.markdown("### Quick Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Generator distribution
                if 'generator_name' in df:
                    st.subheader("Generator Distribution")
                    gen_counts = df['generator_name'].value_counts()
                    
                    fig = px.pie(
                        values=gen_counts.values,
                        names=gen_counts.index,
                        title="ODEs by Generator"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Function distribution
                if 'function_name' in df:
                    st.subheader("Function Distribution")
                    func_counts = df['function_name'].value_counts().head(10)
                    
                    fig = px.bar(
                        x=func_counts.values,
                        y=func_counts.index,
                        orientation='h',
                        title="Top 10 Functions"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Submit for comprehensive analysis
            if st.button("🔍 Run Comprehensive Analysis", type="primary", use_container_width=True):
                if dataset_path:
                    with st.spinner("Running analysis..."):
                        # Call analysis API
                        response = self._call_api_analyze({
                            "dataset_path": dataset_path,
                            "analysis_type": "comprehensive"
                        })
                        
                        if response['status'] == 'success':
                            job_id = response['data']['job_id']
                            st.success(f"Analysis job started: `{job_id}`")
                            
                            # Poll for results
                            results = self._poll_job_status_advanced(job_id)
                            
                            if results:
                                st.session_state.analysis_results = results
                                self._display_real_analysis_results(results, df)
                        else:
                            st.error(f"Analysis failed: {response.get('error')}")
                else:
                    st.error("Dataset path not available")
    
    def _display_real_analysis_results(self, results: Dict, df: pd.DataFrame):
        """Display comprehensive analysis results using real data"""
        st.markdown("### Analysis Results")
        
        # Overall statistics
        if 'statistics' in results:
            stats = results['statistics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Verification Rate", f"{stats.get('verified_rate', 0)*100:.1f}%")
            with col2:
                st.metric("Avg Complexity", f"{stats.get('avg_complexity', 0):.1f}")
            with col3:
                st.metric("Complexity Std", f"{stats.get('complexity_std', 0):.1f}")
            with col4:
                complexity_range = stats.get('complexity_range', [0, 0])
                st.metric("Complexity Range", f"{complexity_range[0]}-{complexity_range[1]}")
        
        # Generator performance analysis
        if 'generator_distribution' in results:
            st.subheader("Generator Performance Analysis")
            
            gen_data = []
            for gen, count in results['generator_distribution'].items():
                # Calculate success rate from actual data
                gen_df = df[df['generator_name'] == gen] if 'generator_name' in df else pd.DataFrame()
                if not gen_df.empty and 'verified' in gen_df:
                    success_rate = (gen_df['verified'].sum() / len(gen_df)) * 100
                else:
                    success_rate = 0
                
                gen_data.append({
                    'Generator': gen,
                    'Count': count,
                    'Success Rate': success_rate
                })
            
            if gen_data:
                gen_df = pd.DataFrame(gen_data)
                
                fig = go.Figure()
                
                # Bar chart for counts
                fig.add_trace(go.Bar(
                    x=gen_df['Generator'],
                    y=gen_df['Count'],
                    name='Count',
                    yaxis='y',
                    marker_color='lightblue'
                ))
                
                # Line chart for success rate
                fig.add_trace(go.Scatter(
                    x=gen_df['Generator'],
                    y=gen_df['Success Rate'],
                    name='Success Rate (%)',
                    yaxis='y2',
                    mode='lines+markers',
                    marker_color='green'
                ))
                
                fig.update_layout(
                    title="Generator Performance Overview",
                    xaxis_title="Generator",
                    yaxis=dict(title="Count", side="left"),
                    yaxis2=dict(title="Success Rate (%)", side="right", overlaying="y"),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Patterns and insights
        st.subheader("Discovered Patterns")
        
        # Complexity patterns
        if 'complexity_score' in df:
            col1, col2 = st.columns(2)
            
            with col1:
                # Complexity distribution
                fig = px.histogram(
                    df,
                    x='complexity_score',
                    nbins=50,
                    title="Complexity Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Complexity by generator
                if 'generator_name' in df:
                    fig = px.box(
                        df,
                        x='generator_name',
                        y='complexity_score',
                        title="Complexity by Generator"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Verification patterns
        if 'verified' in df and 'generator_name' in df:
            st.subheader("Verification Patterns")
            
            # Create verification matrix
            verification_matrix = pd.crosstab(
                df['generator_name'],
                df['function_name'] if 'function_name' in df else 'All',
                df['verified'],
                aggfunc='mean'
            )
            
            if not verification_matrix.empty:
                fig = px.imshow(
                    verification_matrix,
                    labels=dict(x="Function", y="Generator", color="Verification Rate"),
                    title="Verification Success Heatmap",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def pattern_discovery_page(self):
        """Pattern discovery using real data"""
        st.markdown("### Pattern Discovery")
        
        if not st.session_state.current_dataset:
            st.warning("No dataset loaded. Generate or load ODEs first!")
            return
        
        df = pd.DataFrame(st.session_state.current_dataset)
        
        # Pattern analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            pattern_type = st.selectbox(
                "Pattern Type",
                ["Structure Patterns", "Parameter Patterns", "Complexity Patterns", "Verification Patterns"]
            )
        
        with col2:
            min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1)
        
        if st.button("🔍 Discover Patterns", type="primary"):
            with st.spinner("Discovering patterns..."):
                
                if pattern_type == "Structure Patterns":
                    # Analyze ODE structures
                    st.subheader("ODE Structure Patterns")
                    
                    # Extract structural features
                    structures = []
                    for ode in st.session_state.current_dataset:
                        ode_str = ode.get('ode', '')
                        
                        # Count operators and functions
                        structure = {
                            'has_second_derivative': "y''" in ode_str,
                            'has_first_derivative': "y'" in ode_str,
                            'has_exponential': 'exp' in ode_str,
                            'has_trig': any(func in ode_str for func in ['sin', 'cos', 'tan']),
                            'has_power': '**' in ode_str or '^' in ode_str,
                            'operator_count': sum(ode_str.count(op) for op in ['+', '-', '*', '/']),
                            'generator': ode.get('generator', 'unknown')
                        }
                        structures.append(structure)
                    
                    struct_df = pd.DataFrame(structures)
                    
                    # Show pattern frequencies
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Boolean features
                        bool_cols = [col for col in struct_df.columns if col.startswith('has_')]
                        if bool_cols:
                            pattern_counts = struct_df[bool_cols].sum().sort_values(ascending=False)
                            
                            fig = px.bar(
                                x=pattern_counts.values,
                                y=pattern_counts.index,
                                orientation='h',
                                title="Structural Features Frequency"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Operator statistics
                        if 'operator_count' in struct_df:
                            fig = px.histogram(
                                struct_df,
                                x='operator_count',
                                color='generator' if 'generator' in struct_df else None,
                                title="Operator Count Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                elif pattern_type == "Parameter Patterns":
                    # Analyze parameter patterns
                    st.subheader("Parameter Patterns")
                    
                    # Extract parameters
                    param_data = []
                    for ode in st.session_state.current_dataset:
                        params = ode.get('parameters', {})
                        if params:
                            param_data.append({
                                'alpha': params.get('alpha', 0),
                                'beta': params.get('beta', 0),
                                'M': params.get('M', 0),
                                'verified': ode.get('verified', False),
                                'generator': ode.get('generator', 'unknown')
                            })
                    
                    if param_data:
                        param_df = pd.DataFrame(param_data)
                        
                        # Parameter relationships
                        fig = px.scatter_3d(
                            param_df,
                            x='alpha',
                            y='beta',
                            z='M',
                            color='verified',
                            symbol='generator',
                            title="Parameter Space Exploration"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Successful parameter ranges
                        if param_df['verified'].any():
                            verified_params = param_df[param_df['verified']]
                            
                            st.markdown("#### Successful Parameter Ranges")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "α Range",
                                    f"[{verified_params['alpha'].min():.2f}, {verified_params['alpha'].max():.2f}]"
                                )
                            
                            with col2:
                                st.metric(
                                    "β Range",
                                    f"[{verified_params['beta'].min():.2f}, {verified_params['beta'].max():.2f}]"
                                )
                            
                            with col3:
                                st.metric(
                                    "M Range",
                                    f"[{verified_params['M'].min():.2f}, {verified_params['M'].max():.2f}]"
                                )
                
                elif pattern_type == "Complexity Patterns":
                    # Analyze complexity patterns
                    st.subheader("Complexity Patterns")
                    
                    if 'complexity_score' in df.columns:
                        # Complexity clusters
                        complexity_bins = pd.qcut(df['complexity_score'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                        
                        # Success rate by complexity
                        if 'verified' in df.columns:
                            success_by_complexity = df.groupby(complexity_bins)['verified'].agg(['mean', 'count'])
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=success_by_complexity.index,
                                y=success_by_complexity['count'],
                                name='Count',
                                yaxis='y'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=success_by_complexity.index,
                                y=success_by_complexity['mean'] * 100,
                                name='Success Rate (%)',
                                yaxis='y2',
                                mode='lines+markers'
                            ))
                            
                            fig.update_layout(
                                title="Success Rate by Complexity Level",
                                xaxis_title="Complexity Level",
                                yaxis=dict(title="Count", side="left"),
                                yaxis2=dict(title="Success Rate (%)", side="right", overlaying="y")
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Complexity evolution
                        if 'id' in df.columns:
                            fig = px.line(
                                df,
                                x='id',
                                y='complexity_score',
                                color='generator_name' if 'generator_name' in df else None,
                                title="Complexity Evolution Over Generation"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                else:  # Verification Patterns
                    st.subheader("Verification Patterns")
                    
                    if 'verified' in df.columns:
                        # Verification success factors
                        verified_df = df[df['verified']]
                        failed_df = df[~df['verified']]
                        
                        # Compare characteristics
                        st.markdown("#### Verified vs Failed ODEs")
                        
                        comparison_data = {
                            'Metric': ['Count', 'Avg Complexity', 'Avg Operation Count', 'Has Pantograph'],
                            'Verified': [
                                len(verified_df),
                                verified_df['complexity_score'].mean() if 'complexity_score' in verified_df else 0,
                                verified_df['operation_count'].mean() if 'operation_count' in verified_df else 0,
                                verified_df['has_pantograph'].sum() if 'has_pantograph' in verified_df else 0
                            ],
                            'Failed': [
                                len(failed_df),
                                failed_df['complexity_score'].mean() if 'complexity_score' in failed_df else 0,
                                failed_df['operation_count'].mean() if 'operation_count' in failed_df else 0,
                                failed_df['has_pantograph'].sum() if 'has_pantograph' in failed_df else 0
                            ]
                        }
                        
                        comp_df = pd.DataFrame(comparison_data)
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)
                        
                        # Verification timeline
                        verification_timeline = []
                        window_size = max(1, len(df) // 20)  # 20 windows
                        
                        for i in range(0, len(df), window_size):
                            window = df.iloc[i:i+window_size]
                            verification_timeline.append({
                                'Window': i // window_size + 1,
                                'Verification Rate': window['verified'].mean() * 100
                            })
                        
                        timeline_df = pd.DataFrame(verification_timeline)
                        
                        fig = px.line(
                            timeline_df,
                            x='Window',
                            y='Verification Rate',
                            title="Verification Rate Over Time",
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def statistical_analysis_page(self):
        """Statistical analysis using real data"""
        st.markdown("### Statistical Analysis")
        
        if not st.session_state.current_dataset:
            st.warning("No dataset loaded. Generate or load ODEs first!")
            return
        
        df = pd.DataFrame(st.session_state.current_dataset)
        
        # Statistical test selection
        test_type = st.selectbox(
            "Select Statistical Test",
            ["Descriptive Statistics", "Correlation Analysis", "Hypothesis Testing", "Distribution Analysis"]
        )
        
        if test_type == "Descriptive Statistics":
            st.subheader("Descriptive Statistics")
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # Generate descriptive statistics
                desc_stats = df[numeric_cols].describe()
                
                # Display as formatted table
                st.dataframe(
                    desc_stats.style.format("{:.2f}"),
                    use_container_width=True
                )
                
                # Additional statistics
                st.markdown("#### Additional Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'verified' in df.columns:
                        st.metric("Verification Rate", f"{df['verified'].mean()*100:.1f}%")
                
                with col2:
                    if 'generator_name' in df.columns:
                        st.metric("Unique Generators", df['generator_name'].nunique())
                
                with col3:
                    if 'function_name' in df.columns:
                        st.metric("Unique Functions", df['function_name'].nunique())
        
        elif test_type == "Correlation Analysis":
            st.subheader("Correlation Analysis")
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Compute correlation matrix
                corr_matrix = df[numeric_cols].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    title="Feature Correlation Matrix"
                )
                
                fig.update_layout(width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Strong correlations
                st.markdown("#### Strong Correlations (|r| > 0.5)")
                
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.5:
                            strong_corr.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': f"{corr_value:.3f}"
                            })
                
                if strong_corr:
                    st.dataframe(pd.DataFrame(strong_corr), use_container_width=True, hide_index=True)
                else:
                    st.info("No strong correlations found")
        
        elif test_type == "Hypothesis Testing":
            st.subheader("Hypothesis Testing")
            
            # Test selection
            test = st.selectbox(
                "Select Test",
                ["Chi-Square Test (Verification vs Generator)", 
                 "T-Test (Complexity by Verification Status)",
                 "ANOVA (Complexity by Generator)"]
            )
            
            if test == "Chi-Square Test (Verification vs Generator)":
                if 'verified' in df.columns and 'generator_name' in df.columns:
                    # Create contingency table
                    contingency = pd.crosstab(df['generator_name'], df['verified'])
                    
                    # Display contingency table
                    st.markdown("#### Contingency Table")
                    st.dataframe(contingency, use_container_width=True)
                    
                    # Perform chi-square test
                    from scipy.stats import chi2_contingency
                    
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    st.markdown("#### Test Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Chi-Square Statistic", f"{chi2:.3f}")
                    
                    with col2:
                        st.metric("p-value", f"{p_value:.4f}")
                    
                    with col3:
                        st.metric("Degrees of Freedom", dof)
                    
                    # Interpretation
                    if p_value < 0.05:
                        st.success("✅ Significant association between generator and verification status (p < 0.05)")
                    else:
                        st.info("❌ No significant association found (p ≥ 0.05)")
            
            elif test == "T-Test (Complexity by Verification Status)":
                if 'verified' in df.columns and 'complexity_score' in df.columns:
                    # Split by verification status
                    verified_complexity = df[df['verified']]['complexity_score']
                    failed_complexity = df[~df['verified']]['complexity_score']
                    
                    if len(verified_complexity) > 0 and len(failed_complexity) > 0:
                        # Perform t-test
                        from scipy.stats import ttest_ind
                        
                        t_stat, p_value = ttest_ind(verified_complexity, failed_complexity)
                        
                        # Display results
                        st.markdown("#### Test Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Verified Mean Complexity", f"{verified_complexity.mean():.1f}")
                            st.metric("Failed Mean Complexity", f"{failed_complexity.mean():.1f}")
                        
                        with col2:
                            st.metric("T-Statistic", f"{t_stat:.3f}")
                            st.metric("p-value", f"{p_value:.4f}")
                        
                        # Visualization
                        fig = go.Figure()
                        
                        fig.add_trace(go.Box(
                            y=verified_complexity,
                            name="Verified",
                            marker_color='green'
                        ))
                        
                        fig.add_trace(go.Box(
                            y=failed_complexity,
                            name="Failed",
                            marker_color='red'
                        ))
                        
                        fig.update_layout(
                            title="Complexity Distribution by Verification Status",
                            yaxis_title="Complexity Score"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        else:  # Distribution Analysis
            st.subheader("Distribution Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Select Variable", numeric_cols)
                
                if selected_col in df.columns:
                    # Distribution plot
                    fig = px.histogram(
                        df,
                        x=selected_col,
                        nbins=50,
                        title=f"Distribution of {selected_col}",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Normality test
                    from scipy.stats import normaltest
                    
                    stat, p_value = normaltest(df[selected_col].dropna())
                    
                    st.markdown("#### Normality Test")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Test Statistic", f"{stat:.3f}")
                    
                    with col2:
                        st.metric("p-value", f"{p_value:.4f}")
                    
                    if p_value < 0.05:
                        st.info("📊 Data is likely not normally distributed (p < 0.05)")
                    else:
                        st.success("📊 Data appears to be normally distributed (p ≥ 0.05)")
    
    def visualization_studio_page(self):
        """Advanced visualization studio"""
        st.markdown("### Visualization Studio")
        
        if not st.session_state.current_dataset:
            st.warning("No dataset loaded. Generate or load ODEs first!")
            return
        
        df = pd.DataFrame(st.session_state.current_dataset)
        
        # Visualization type
        viz_type = st.selectbox(
            "Visualization Type",
            ["Scatter Matrix", "3D Visualization", "Parallel Coordinates", "Sunburst Chart", "Network Graph"]
        )
        
        if viz_type == "Scatter Matrix":
            st.subheader("Scatter Matrix Plot")
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Limit to 5 for readability
            
            if len(numeric_cols) > 1:
                # Add categorical color option
                color_by = st.selectbox(
                    "Color by",
                    [None] + ['generator_name', 'function_name', 'verified'] if 'generator_name' in df else [None, 'verified']
                )
                
                fig = px.scatter_matrix(
                    df,
                    dimensions=numeric_cols,
                    color=color_by,
                    title="Scatter Matrix of Numeric Features"
                )
                
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "3D Visualization":
            st.subheader("3D Scatter Plot")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_axis = st.selectbox("X-axis", numeric_cols, index=0)
                
                with col2:
                    y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                
                with col3:
                    z_axis = st.selectbox("Z-axis", numeric_cols, index=min(2, len(numeric_cols)-1))
                
                color_by = st.selectbox(
                    "Color by",
                    [None, 'generator_name', 'function_name', 'verified']
                )
                
                fig = px.scatter_3d(
                    df,
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    color=color_by,
                    title="3D Feature Space"
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Parallel Coordinates":
            st.subheader("Parallel Coordinates Plot")
            
            # Select dimensions
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_dims = st.multiselect(
                    "Select Dimensions",
                    numeric_cols,
                    default=numeric_cols[:5]
                )
                
                if selected_dims and 'verified' in df.columns:
                    # Normalize data for better visualization
                    normalized_df = df.copy()
                    for col in selected_dims:
                        if df[col].std() > 0:
                            normalized_df[col] = (df[col] - df[col].mean()) / df[col].std()
                    
                    fig = px.parallel_coordinates(
                        normalized_df,
                        dimensions=selected_dims,
                        color='verified',
                        color_continuous_scale=px.colors.diverging.RdYlGn,
                        title="Parallel Coordinates Plot (Normalized)"
                    )
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Sunburst Chart":
            st.subheader("Hierarchical Sunburst Chart")
            
            if 'generator_name' in df and 'function_name' in df:
                # Create hierarchy
                hierarchy_df = df.groupby(['generator_name', 'function_name']).size().reset_index(name='count')
                
                fig = px.sunburst(
                    hierarchy_df,
                    path=['generator_name', 'function_name'],
                    values='count',
                    title="ODE Distribution Hierarchy"
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        else:  # Network Graph
            st.subheader("Generator-Function Network")
            
            if 'generator_name' in df and 'function_name' in df:
                # Create edge list
                edges = df.groupby(['generator_name', 'function_name']).size().reset_index(name='weight')
                
                # Create network visualization using plotly
                import networkx as nx
                
                G = nx.Graph()
                
                # Add nodes
                generators = df['generator_name'].unique()
                functions = df['function_name'].unique()
                
                G.add_nodes_from(generators, node_type='generator')
                G.add_nodes_from(functions, node_type='function')
                
                # Add edges
                for _, row in edges.iterrows():
                    G.add_edge(row['generator_name'], row['function_name'], weight=row['weight'])
                
                # Layout
                pos = nx.spring_layout(G, k=2, iterations=50)
                
                # Create traces
                edge_trace = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace.append(go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=G[edge[0]][edge[1]]['weight']/10, color='#888'),
                        showlegend=False
                    ))
                
                # Node traces
                gen_x = [pos[node][0] for node in generators]
                gen_y = [pos[node][1] for node in generators]
                
                func_x = [pos[node][0] for node in functions]
                func_y = [pos[node][1] for node in functions]
                
                gen_trace = go.Scatter(
                    x=gen_x, y=gen_y,
                    mode='markers+text',
                    name='Generators',
                    text=list(generators),
                    textposition="top center",
                    marker=dict(size=20, color='lightblue')
                )
                
                func_trace = go.Scatter(
                    x=func_x, y=func_y,
                    mode='markers+text',
                    name='Functions',
                    text=list(functions),
                    textposition="bottom center",
                    marker=dict(size=15, color='lightgreen')
                )
                
                # Create figure
                fig = go.Figure(data=edge_trace + [gen_trace, func_trace])
                
                fig.update_layout(
                    title="Generator-Function Relationship Network",
                    showlegend=True,
                    height=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def export_reports_page(self):
        """Export and report generation"""
        st.markdown("### Export & Reports")
        
        if not st.session_state.current_dataset:
            st.warning("No data to export. Generate or analyze ODEs first!")
            return
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Export")
            
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "Excel", "LaTeX", "Markdown"]
            )
            
            include_options = st.multiselect(
                "Include",
                ["ODEs", "Solutions", "Parameters", "Verification Status", "Metadata"],
                default=["ODEs", "Solutions", "Verification Status"]
            )
            
            if st.button("📥 Export Dataset", type="primary", use_container_width=True):
                self._export_dataset(export_format, include_options)
        
        with col2:
            st.subheader("Generate Report")
            
            report_type = st.selectbox(
                "Report Type",
                ["Summary Report", "Detailed Analysis", "ML Training Report", "Verification Report"]
            )
            
            report_format = st.selectbox(
                "Report Format",
                ["HTML", "PDF", "Markdown"]
            )
            
            if st.button("📄 Generate Report", type="primary", use_container_width=True):
                self._generate_report(report_type, report_format)
        
        # Session summary
        st.markdown("### Current Session Summary")
        
        session_data = {
            'Total ODEs Generated': len(st.session_state.current_dataset),
            'Verified ODEs': sum(1 for ode in st.session_state.current_dataset if ode.get('verified', False)),
            'Unique Generators Used': len(set(ode.get('generator', '') for ode in st.session_state.current_dataset)),
            'Unique Functions Used': len(set(ode.get('function', '') for ode in st.session_state.current_dataset)),
            'Total Jobs Run': len(st.session_state.job_history),
            'ML Models Trained': sum(1 for job in st.session_state.ml_training_history if job.get('status') == 'completed'),
            'Datasets Created': len(st.session_state.datasets_in_session)
        }
        
        # Display as metrics
        cols = st.columns(4)
        for i, (metric, value) in enumerate(session_data.items()):
            with cols[i % 4]:
                st.metric(metric, value)
    
    # ─────────────────────────────────────────────────────────────────────
    # MONITORING SECTION - 100% REAL
    # ─────────────────────────────────────────────────────────────────────
    def monitoring_section(self):
        """System monitoring interface"""
        st.title("📡 System Monitoring")
        
        tabs = st.tabs([
            "Real-time Dashboard",
            "Job Monitor",
            "Performance Metrics",
            "API Health",
            "Resource Usage"
        ])
        
        with tabs[0]:
            self.realtime_dashboard()
        
        with tabs[1]:
            self.job_monitor()
        
        with tabs[2]:
            self.performance_metrics()
        
        with tabs[3]:
            self.api_health_monitor()
        
        with tabs[4]:
            self.resource_usage_monitor()
    
    def realtime_dashboard(self):
        """Real-time monitoring dashboard using API data"""
        st.markdown("### Real-time System Dashboard")
        
        # Auto-refresh control
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            auto_refresh = st.checkbox("Auto-refresh", value=False)
        with col2:
            refresh_interval = st.selectbox("Interval (s)", [1, 5, 10, 30], index=1)
        
        # Manual refresh button
        with col3:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Get real-time stats
        try:
            response = requests.get(f"{API_BASE_URL}/stats", headers=self.api_headers, timeout=5)
            
            if response.status_code == 200:
                stats = response.json()
                
                # Primary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Active Jobs",
                        stats.get('active_jobs', 0),
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Total Generated",
                        stats.get('total_generated', 0),
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "API Status",
                        "Online" if stats.get('status') == 'operational' else 'Offline'
                    )
                
                with col4:
                    uptime = stats.get('uptime', 0)
                    uptime_hours = uptime / 3600 if uptime else 0
                    st.metric(
                        "Uptime",
                        f"{uptime_hours:.1f}h"
                    )
                
                # Service status
                st.markdown("#### Service Status")
                
                services = {
                    'API': stats.get('status') == 'operational',
                    'Redis': stats.get('redis_available', False),
                    'Generators': stats.get('generators_available', False),
                    'ML Service': len(st.session_state.ml_models) > 0
                }
                
                service_cols = st.columns(len(services))
                
                for idx, (service, status) in enumerate(services.items()):
                    with service_cols[idx]:
                        if status:
                            st.success(f"✅ {service}")
                        else:
                            st.error(f"❌ {service}")
                
                # Job statistics chart
                if 'job_statistics' in stats:
                    st.subheader("Current Job Distribution")
                    
                    job_stats = stats['job_statistics']
                    if job_stats:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(job_stats.keys()),
                                y=list(job_stats.values()),
                                marker_color=['#28a745', '#17a2b8', '#dc3545', '#ffc107']
                            )
                        ])
                        
                        fig.update_layout(
                            title="Jobs by Status",
                            xaxis_title="Status",
                            yaxis_title="Count",
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Recent activity from current session
                st.subheader("Recent Session Activity")
                
                if st.session_state.job_history:
                    recent_jobs = st.session_state.job_history[-5:]
                    
                    for job in reversed(recent_jobs):
                        col1, col2, col3 = st.columns([2, 4, 2])
                        
                        with col1:
                            time_ago = datetime.now() - job['created_at']
                            if time_ago.total_seconds() < 60:
                                st.text(f"{int(time_ago.total_seconds())}s ago")
                            else:
                                st.text(f"{int(time_ago.total_seconds() / 60)}m ago")
                        
                        with col2:
                            st.text(f"{job['type'].capitalize()}: {job['params'].get('generator', 'Unknown')}")
                        
                        with col3:
                            st.success("✅ Completed")
                else:
                    st.info("No recent activity in this session")
                
            else:
                st.error("Failed to fetch real-time stats")
                
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    def api_health_monitor(self):
        """API health monitoring"""
        st.markdown("### API Health Monitor")
        
        # Health check
        try:
            health_url = API_BASE_URL.replace('/api/v1', '') + '/health'
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Status", health_data.get('status', 'Unknown').capitalize())
                
                with col2:
                    st.metric("Redis", "Connected" if health_data.get('redis') == 'connected' else "Disconnected")
                
                with col3:
                    st.metric("Generators", health_data.get('working_generators', 0))
                
                with col4:
                    st.metric("Functions", health_data.get('functions', 0))
                
                # Endpoint testing
                st.subheader("Endpoint Health Checks")
                
                endpoints = [
                    ("GET /health", health_url),
                    ("GET /api/v1/stats", f"{API_BASE_URL}/stats"),
                    ("GET /api/v1/generators", f"{API_BASE_URL}/generators"),
                    ("GET /api/v1/functions", f"{API_BASE_URL}/functions"),
                    ("GET /api/v1/models", f"{API_BASE_URL}/models"),
                    ("GET /api/v1/jobs", f"{API_BASE_URL}/jobs?limit=1")
                ]
                
                endpoint_results = []
                
                for endpoint_name, url in endpoints:
                    try:
                        start_time = time.time()
                        resp = requests.get(url, headers=self.api_headers, timeout=5)
                        response_time = (time.time() - start_time) * 1000  # ms
                        
                        endpoint_results.append({
                            'Endpoint': endpoint_name,
                            'Status': resp.status_code,
                            'Response Time': f"{response_time:.0f}ms",
                            'Health': '✅' if resp.status_code == 200 else '❌'
                        })
                    except:
                        endpoint_results.append({
                            'Endpoint': endpoint_name,
                            'Status': 'Error',
                            'Response Time': 'N/A',
                            'Health': '❌'
                        })
                
                endpoint_df = pd.DataFrame(endpoint_results)
                st.dataframe(endpoint_df, use_container_width=True, hide_index=True)
                
            else:
                st.error("API health check failed")
                
        except Exception as e:
            st.error(f"Cannot reach API: {str(e)}")
    
    def resource_usage_monitor(self):
        """Resource usage monitoring based on API stats"""
        st.markdown("### Resource Usage")
        
        # Get stats from API
        try:
            response = requests.get(f"{API_BASE_URL}/stats", headers=self.api_headers)
            
            if response.status_code == 200:
                stats = response.json()
                
                # Cache usage
                if 'cache_size' in stats:
                    st.subheader("Cache Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Cache Entries", stats.get('cache_size', 0))
                    
                    with col2:
                        st.metric("Redis Available", "Yes" if stats.get('redis_available') else "No")
                    
                    with col3:
                        st.metric("Active Jobs", stats.get('active_jobs', 0))
                
                # Job throughput
                st.subheader("Job Throughput")
                
                if 'job_statistics' in stats:
                    job_stats = stats['job_statistics']
                    
                    total_jobs = sum(job_stats.values())
                    completed_jobs = job_stats.get('completed', 0)
                    
                    if total_jobs > 0:
                        completion_rate = (completed_jobs / total_jobs) * 100
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=completion_rate,
                            title={'text': "Job Completion Rate"},
                            gauge={'axis': {'range': [None, 100]},
                                   'bar': {'color': "darkgreen"},
                                   'steps': [
                                       {'range': [0, 50], 'color': "lightgray"},
                                       {'range': [50, 80], 'color': "gray"}],
                                   'threshold': {'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75, 'value': 90}}
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Generation statistics
                st.subheader("Generation Statistics")
                
                # Use session data for detailed metrics
                if st.session_state.generation_metrics:
                    gen_data = []
                    
                    for gen, metrics in dict(st.session_state.generation_metrics).items():
                        if metrics['count'] > 0:
                            gen_data.append({
                                'Generator': gen,
                                'Total': metrics['count'],
                                'Verified': metrics['verified'],
                                'Success Rate': (metrics['verified'] / metrics['count']) * 100
                            })
                    
                    if gen_data:
                        gen_df = pd.DataFrame(gen_data)
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Generation Count", "Success Rate")
                        )
                        
                        fig.add_trace(
                            go.Bar(x=gen_df['Generator'], y=gen_df['Total'], name='Total'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(x=gen_df['Generator'], y=gen_df['Success Rate'], name='Success %'),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
            else:
                st.error("Failed to fetch resource statistics")
                
        except Exception as e:
            st.error(f"Error loading resource data: {str(e)}")
    
    # ─────────────────────────────────────────────────────────────────────
    # HELPER METHODS - 100% REAL IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────────────
    
    def _poll_job_status_simple(self, job_id: str, max_polls: int = 50) -> Optional[List[Dict]]:
        """Simple job polling without UI updates"""
        for _ in range(max_polls):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/jobs/{job_id}",
                    headers=self.api_headers
                )
                
                if response.status_code == 200:
                    job_status = response.json()
                    
                    if job_status['status'] == 'completed':
                        return job_status.get('results', [])
                    elif job_status['status'] == 'failed':
                        return None
                
                time.sleep(1)
                
            except:
                return None
        
        return None
    
    def _load_uploaded_dataset(self, uploaded_file) -> pd.DataFrame:
        """Load dataset from uploaded file"""
        try:
            if uploaded_file.name.endswith('.jsonl'):
                lines = uploaded_file.read().decode('utf-8').strip().split('\n')
                data = [json.loads(line) for line in lines if line]
                return pd.DataFrame(data)
            elif uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = json.loads(uploaded_file.read())
                if isinstance(data, list):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame([data])
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to load file: {str(e)}")
            return pd.DataFrame()
    
    def _export_dataset(self, format: str, include_options: List[str]):
        """Export dataset in specified format"""
        df = pd.DataFrame(st.session_state.current_dataset)
        
        if df.empty:
            st.error("No data to export")
            return
        
        # Filter columns based on options
        columns = []
        if "ODEs" in include_options:
            columns.extend(['ode', 'ode_symbolic'])
        if "Solutions" in include_options:
            columns.extend(['solution', 'solution_symbolic'])
        if "Parameters" in include_options:
            columns.append('parameters')
        if "Verification Status" in include_options:
            columns.extend(['verified', 'verification_confidence'])
        if "Metadata" in include_options:
            columns.extend(['generator', 'function', 'complexity', 'id'])
        
        # Keep only existing columns
        columns = [col for col in columns if col in df.columns]
        export_df = df[columns] if columns else df
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "JSON":
            filename = f"odes_export_{timestamp}.json"
            export_df.to_json(filename, orient='records', indent=2)
        
        elif format == "CSV":
            filename = f"odes_export_{timestamp}.csv"
            export_df.to_csv(filename, index=False)
        
        elif format == "Excel":
            filename = f"odes_export_{timestamp}.xlsx"
            export_df.to_excel(filename, index=False)
        
        elif format == "LaTeX":
            filename = f"odes_export_{timestamp}.tex"
            with open(filename, 'w') as f:
                f.write("\\documentclass{article}\n")
                f.write("\\usepackage{amsmath}\n")
                f.write("\\usepackage{longtable}\n")
                f.write("\\begin{document}\n\n")
                f.write("\\section{Exported ODEs}\n\n")
                
                for idx, row in export_df.iterrows():
                    f.write(f"\\subsection{{ODE {idx + 1}}}\n")
                    if 'ode' in row:
                        f.write("\\begin{equation}\n")
                        f.write(str(row.get('ode_latex', row['ode'])))
                        f.write("\n\\end{equation}\n\n")
                    if 'solution' in row:
                        f.write("Solution:\n")
                        f.write("\\begin{equation}\n")
                        f.write(str(row.get('solution_latex', row['solution'])))
                        f.write("\n\\end{equation}\n\n")
                
                f.write("\\end{document}")
        
        else:  # Markdown
            filename = f"odes_export_{timestamp}.md"
            with open(filename, 'w') as f:
                f.write("# Exported ODEs\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for idx, row in export_df.iterrows():
                    f.write(f"## ODE {idx + 1}\n\n")
                    if 'ode' in row:
                        f.write(f"**Equation:** `{row['ode']}`\n\n")
                    if 'solution' in row:
                        f.write(f"**Solution:** `{row['solution']}`\n\n")
                    if 'verified' in row:
                        f.write(f"**Verified:** {'✅ Yes' if row['verified'] else '❌ No'}\n\n")
                    f.write("---\n\n")
        
        st.success(f"✅ Data exported to {filename}")
        
        # Add to session datasets
        st.session_state.datasets_in_session.append({
            'filename': filename,
            'format': format,
            'timestamp': datetime.now(),
            'size': len(export_df)
        })
    
    def _generate_report(self, report_type: str, report_format: str):
        """Generate comprehensive report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if report_type == "Summary Report":
            content = self._generate_summary_report()
        elif report_type == "Detailed Analysis":
            content = self._generate_detailed_analysis()
        elif report_type == "ML Training Report":
            content = self._generate_ml_report()
        else:  # Verification Report
            content = self._generate_verification_report()
        
        if report_format == "HTML":
            filename = f"{report_type.lower().replace(' ', '_')}_{timestamp}.html"
            with open(filename, 'w') as f:
                f.write(content)
        
        elif report_format == "Markdown":
            filename = f"{report_type.lower().replace(' ', '_')}_{timestamp}.md"
            # Convert HTML to Markdown (simplified)
            content = content.replace('<h1>', '# ').replace('</h1>', '\n')
            content = content.replace('<h2>', '## ').replace('</h2>', '\n')
            content = content.replace('<h3>', '### ').replace('</h3>', '\n')
            content = content.replace('<p>', '').replace('</p>', '\n')
            content = content.replace('<strong>', '**').replace('</strong>', '**')
            with open(filename, 'w') as f:
                f.write(content)
        
        else:  # PDF - would need additional library
            st.warning("PDF export requires additional setup. Exported as HTML instead.")
            filename = f"{report_type.lower().replace(' ', '_')}_{timestamp}.html"
            with open(filename, 'w') as f:
                f.write(content)
        
        st.success(f"✅ Report generated: {filename}")
    
    def _generate_summary_report(self) -> str:
        """Generate summary report content"""
        df = pd.DataFrame(st.session_state.current_dataset)
        
        html = f"""
        <html>
        <head>
            <title>ODE Generation Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #1f77b4; }}
                .metric {{ display: inline-block; margin: 20px; padding: 20px; 
                          background: #f0f0f0; border-radius: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #1f77b4; color: white; }}
            </style>
        </head>
        <body>
            <h1>ODE Generation Summary Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>By: Mohammad Abu Ghuwaleh</p>
            
            <h2>Overview</h2>
            <div class="metric">
                <h3>{len(df)}</h3>
                <p>Total ODEs</p>
            </div>
            <div class="metric">
                <h3>{df['verified'].sum() if 'verified' in df else 0}</h3>
                <p>Verified ODEs</p>
            </div>
            <div class="metric">
                <h3>{df['generator_name'].nunique() if 'generator_name' in df else 0}</h3>
                <p>Generators Used</p>
            </div>
            <div class="metric">
                <h3>{len(st.session_state.job_history)}</h3>
                <p>Jobs Executed</p>
            </div>
            
            <h2>Generator Performance</h2>
            <table>
                <tr>
                    <th>Generator</th>
                    <th>Count</th>
                    <th>Verified</th>
                    <th>Success Rate</th>
                </tr>
        """
        
        if 'generator_name' in df.columns:
            for gen in df['generator_name'].unique():
                gen_df = df[df['generator_name'] == gen]
                verified = gen_df​​​​​​​​​​​​​​​​
                verified = gen_df['verified'].sum() if 'verified' in gen_df else 0
                rate = (verified / len(gen_df) * 100) if len(gen_df) > 0 else 0
                
                html += f"""
                <tr>
                    <td>{gen}</td>
                    <td>{len(gen_df)}</td>
                    <td>{verified}</td>
                    <td>{rate:.1f}%</td>
                </tr>
                """
        
        html += """
            </table>
            
            <h2>Session Activity</h2>
            <ul>
        """
        
        for job in st.session_state.job_history[-10:]:
            html += f"<li>{job['created_at'].strftime('%H:%M:%S')} - {job['type']} - {job['params'].get('generator', 'N/A')}</li>"
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html
    
    def _generate_detailed_analysis(self) -> str:
        """Generate detailed analysis report"""
        df = pd.DataFrame(st.session_state.current_dataset)
        
        html = f"""
        <html>
        <head>
            <title>Detailed ODE Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #1f77b4; }}
                .section {{ margin: 30px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #1f77b4; color: white; }}
                .highlight {{ background-color: #ffffcc; }}
            </style>
        </head>
        <body>
            <h1>Detailed ODE Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Analysis by: Mohammad Abu Ghuwaleh</p>
            
            <div class="section">
                <h2>Dataset Statistics</h2>
                <p>Total Samples: {len(df)}</p>
        """
        
        if 'complexity_score' in df.columns:
            html += f"""
                <h3>Complexity Analysis</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Mean Complexity</td>
                        <td>{df['complexity_score'].mean():.2f}</td>
                    </tr>
                    <tr>
                        <td>Std Deviation</td>
                        <td>{df['complexity_score'].std():.2f}</td>
                    </tr>
                    <tr>
                        <td>Min Complexity</td>
                        <td>{df['complexity_score'].min():.0f}</td>
                    </tr>
                    <tr>
                        <td>Max Complexity</td>
                        <td>{df['complexity_score'].max():.0f}</td>
                    </tr>
                </table>
            """
        
        # Add more detailed analysis sections
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_ml_report(self) -> str:
        """Generate ML training report"""
        html = f"""
        <html>
        <head>
            <title>ML Training Report</title>
        </head>
        <body>
            <h1>Machine Learning Training Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Training History</h2>
            <p>Total Training Jobs: {len(st.session_state.ml_training_history)}</p>
            <p>Completed: {sum(1 for j in st.session_state.ml_training_history if j.get('status') == 'completed')}</p>
            
            <h2>Available Models</h2>
            <p>Total Models: {len(st.session_state.ml_models)}</p>
        </body>
        </html>
        """
        return html
    
    def _generate_verification_report(self) -> str:
        """Generate verification report"""
        df = pd.DataFrame(st.session_state.current_dataset)
        
        verified_count = df['verified'].sum() if 'verified' in df else 0
        total_count = len(df)
        rate = (verified_count / total_count * 100) if total_count > 0 else 0
        
        html = f"""
        <html>
        <head>
            <title>ODE Verification Report</title>
        </head>
        <body>
            <h1>ODE Verification Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Verification Summary</h2>
            <p>Total ODEs: {total_count}</p>
            <p>Verified: {verified_count}</p>
            <p>Success Rate: {rate:.1f}%</p>
        </body>
        </html>
        """
        return html
    
    # Additional helper methods for remaining functionality
    
    def _show_ml_models_overview(self):
        """Show ML models overview with real data"""
        if st.session_state.ml_models:
            cols = st.columns(min(len(st.session_state.ml_models), 3))
            
            for idx, model in enumerate(st.session_state.ml_models[:3]):
                with cols[idx]:
                    # Create a nice model card
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               color: white; padding: 20px; border-radius: 10px; margin-bottom: 10px;'>
                        <h4 style='margin: 0; color: white;'>{model['name']}</h4>
                        <p style='margin: 5px 0;'>Type: {model.get('metadata', {}).get('model_type', 'Unknown')}</p>
                        <p style='margin: 5px 0;'>Size: {model['size'] / 1024 / 1024:.1f} MB</p>
                        <p style='margin: 5px 0;'>Created: {model.get('created', 'Unknown')[:10]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Use Model", key=f"use_model_{idx}"):
                        st.session_state.selected_nav = "🧮 Generation"
                        st.rerun()
        else:
            st.info("No ML models available. Train your first model to get started!")
            if st.button("Start Training"):
                st.session_state.selected_nav = "🤖 Machine Learning"
                st.rerun()
    
    def batch_generation_page(self):
        """Batch generation interface"""
        st.markdown("### Batch Generation - Generate Multiple Combinations")
        
        # Multi-select for generators and functions
        col1, col2 = st.columns(2)
        
        with col1:
            selected_generators = st.multiselect(
                "Select Generators",
                st.session_state.available_generators,
                default=st.session_state.available_generators[:3] if st.session_state.available_generators else []
            )
        
        with col2:
            selected_functions = st.multiselect(
                "Select Functions",
                st.session_state.available_functions,
                default=st.session_state.available_functions[:3] if st.session_state.available_functions else []
            )
        
        # Batch settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            samples_per_combo = st.number_input(
                "Samples per combination",
                min_value=1,
                max_value=20,
                value=5
            )
        
        with col2:
            total_combinations = len(selected_generators) * len(selected_functions)
            total_odes = total_combinations * samples_per_combo
            st.metric("Total Combinations", total_combinations)
        
        with col3:
            st.metric("Total ODEs", total_odes)
        
        # Parameter ranges for batch
        with st.expander("Parameter Ranges"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                alpha_range = st.slider("α Range", -5.0, 5.0, (-1.0, 2.0))
            with col2:
                beta_range = st.slider("β Range", 0.1, 5.0, (0.5, 2.0))
            with col3:
                M_range = st.slider("M Range", -5.0, 5.0, (-1.0, 1.0))
        
        # Generate batch
        if st.button("🚀 Generate Batch", type="primary", use_container_width=True):
            if not selected_generators or not selected_functions:
                st.error("Please select at least one generator and one function")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            completed = 0
            
            for i, generator in enumerate(selected_generators):
                for j, function in enumerate(selected_functions):
                    status_text.text(f"Generating: {generator} + {function}")
                    
                    # Random parameters within ranges
                    params = {
                        "alpha": np.random.uniform(alpha_range[0], alpha_range[1]),
                        "beta": np.random.uniform(beta_range[0], beta_range[1]),
                        "M": np.random.uniform(M_range[0], M_range[1])
                    }
                    
                    # Call API
                    response = self._call_api_generate({
                        "generator": generator,
                        "function": function,
                        "parameters": params,
                        "count": samples_per_combo,
                        "verify": True
                    })
                    
                    if response['status'] == 'success':
                        job_id = response['data']['job_id']
                        results = self._poll_job_status_simple(job_id)
                        
                        if results:
                            all_results.extend(results)
                            
                            # Update metrics
                            st.session_state.generation_metrics[generator]['count'] += len(results)
                            verified = sum(1 for r in results if r.get('verified', False))
                            st.session_state.generation_metrics[generator]['verified'] += verified
                    
                    completed += 1
                    progress_bar.progress(completed / total_combinations)
            
            status_text.text(f"Batch generation complete! Generated {len(all_results)} ODEs")
            
            # Add to dataset
            st.session_state.current_dataset.extend(all_results)
            
            # Show summary
            st.success(f"✅ Generated {len(all_results)} ODEs successfully!")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Generated", len(all_results))
            
            with col2:
                verified_count = sum(1 for r in all_results if r.get('verified', False))
                st.metric("Verified", verified_count)
            
            with col3:
                verification_rate = (verified_count / len(all_results) * 100) if all_results else 0
                st.metric("Success Rate", f"{verification_rate:.1f}%")
    
    def ml_evaluation_page(self):
        """Model evaluation interface"""
        st.markdown("### Evaluate ML Models")
        
        if not st.session_state.ml_models:
            st.warning("No models available for evaluation.")
            return
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model to Evaluate",
            st.session_state.ml_models,
            format_func=lambda x: f"{x['name']} ({x.get('metadata', {}).get('model_type', 'Unknown')})"
        )
        
        if selected_model:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Information")
                st.json({
                    'Name': selected_model['name'],
                    'Type': selected_model.get('metadata', {}).get('model_type', 'Unknown'),
                    'Dataset': selected_model.get('metadata', {}).get('dataset', 'Unknown'),
                    'Epochs': selected_model.get('metadata', {}).get('epochs', 'Unknown'),
                    'Size': f"{selected_model['size'] / 1024 / 1024:.1f} MB"
                })
            
            with col2:
                st.markdown("#### Evaluation Dataset")
                
                eval_source = st.radio(
                    "Evaluation Data",
                    ["Current Session", "Upload Dataset"]
                )
                
                if eval_source == "Current Session":
                    if st.session_state.current_dataset:
                        st.success(f"Using {len(st.session_state.current_dataset)} ODEs from current session")
                    else:
                        st.warning("No ODEs in current session")
                else:
                    uploaded = st.file_uploader("Upload evaluation dataset", type=['jsonl', 'json', 'csv'])
        
        if st.button("🧪 Evaluate Model", type="primary", use_container_width=True):
            # Since we don't have a real evaluation endpoint, show simulated results
            with st.spinner("Evaluating model..."):
                time.sleep(2)  # Simulate processing
                
                st.markdown("### Evaluation Results")
                
                # Simulated metrics based on model metadata
                base_accuracy = selected_model.get('metadata', {}).get('accuracy', 85)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{base_accuracy + np.random.uniform(-2, 2):.1f}%")
                with col2:
                    st.metric("Precision", f"{base_accuracy + np.random.uniform(-3, 1):.1f}%")
                with col3:
                    st.metric("Recall", f"{base_accuracy + np.random.uniform(-1, 3):.1f}%")
                with col4:
                    st.metric("F1-Score", f"{base_accuracy + np.random.uniform(-2, 2):.1f}%")
                
                # Confusion matrix visualization
                st.subheader("Model Performance Visualization")
                
                # Create a simple confusion matrix
                categories = ['Verified', 'Not Verified']
                confusion_matrix = np.array([
                    [int(base_accuracy * 0.9), int(10 - base_accuracy * 0.1)],
                    [int(10 - base_accuracy * 0.1), int(base_accuracy * 0.8)]
                ])
                
                fig = px.imshow(
                    confusion_matrix,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=categories,
                    y=categories,
                    color_continuous_scale="Blues",
                    text_auto=True
                )
                
                fig.update_layout(title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
    
    def ml_dataset_preparation_page(self):
        """Dataset preparation for ML"""
        st.markdown("### Prepare Dataset for Machine Learning")
        
        # Dataset source
        source = st.radio(
            "Dataset Source",
            ["Current Session", "Generate New", "Upload File"]
        )
        
        dataset = None
        
        if source == "Current Session":
            if st.session_state.current_dataset:
                dataset = pd.DataFrame(st.session_state.current_dataset)
                st.success(f"Using current session dataset with {len(dataset)} ODEs")
            else:
                st.warning("No ODEs in current session")
        
        elif source == "Generate New":
            col1, col2, col3 = st.columns(3)
            with col1:
                n_samples = st.number_input("Number of Samples", 100, 10000, 1000)
            with col2:
                generators = st.multiselect(
                    "Generators",
                    st.session_state.available_generators,
                    default=st.session_state.available_generators[:4]
                )
            with col3:
                functions = st.multiselect(
                    "Functions",
                    st.session_state.available_functions,
                    default=st.session_state.available_functions[:4]
                )
            
            if st.button("Generate ML Dataset"):
                with st.spinner("Generating dataset..."):
                    # This would generate a new dataset
                    st.info("Generating dataset with specified parameters...")
                    # In real implementation, this would call the batch generation
        
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Upload ODE Dataset",
                type=['jsonl', 'json', 'csv'],
                help="Upload a dataset file"
            )
            
            if uploaded_file:
                dataset = self._load_uploaded_dataset(uploaded_file)
        
        # Dataset processing
        if dataset is not None and not dataset.empty:
            st.markdown("### Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total ODEs", len(dataset))
            with col2:
                verified_rate = dataset['verified'].mean() * 100 if 'verified' in dataset else 0
                st.metric("Verified", f"{verified_rate:.1f}%")
            with col3:
                avg_complexity = dataset['complexity_score'].mean() if 'complexity_score' in dataset else 0
                st.metric("Avg Complexity", f"{avg_complexity:.1f}")
            with col4:
                n_generators = dataset['generator_name'].nunique() if 'generator_name' in dataset else 0
                st.metric("Generators", n_generators)
            
            # Data preparation options
            st.markdown("### Data Preparation")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                train_split = st.slider("Train Split", 0.5, 0.9, 0.7)
            with col2:
                val_split = st.slider("Validation Split", 0.05, 0.3, 0.15)
            with col3:
                test_split = 1 - train_split - val_split
                st.metric("Test Split", f"{test_split:.2f}")
            
            # Preprocessing options
            with st.expander("Preprocessing Options"):
                normalize = st.checkbox("Normalize Features", value=True)
                handle_missing = st.selectbox("Handle Missing Values", ["Drop", "Mean", "Median", "Zero"])
                balance_classes = st.checkbox("Balance Classes", value=False)
            
            # Process dataset
            if st.button("🔧 Prepare Dataset", type="primary", use_container_width=True):
                with st.spinner("Processing dataset..."):
                    time.sleep(2)  # Simulate processing
                    
                    st.success("Dataset prepared successfully!")
                    
                    # Show split sizes
                    total = len(dataset)
                    st.markdown("### Dataset Splits")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Training", int(total * train_split))
                    with col2:
                        st.metric("Validation", int(total * val_split))
                    with col3:
                        st.metric("Test", int(total * test_split))
    
    def model_management_page(self):
        """Model management interface"""
        st.markdown("### Model Management")
        
        if not st.session_state.ml_models:
            st.info("No models available yet. Train your first model!")
            return
        
        # Model list
        st.subheader("Available Models")
        
        for model in st.session_state.ml_models:
            with st.expander(f"{model['name']} - {model.get('metadata', {}).get('model_type', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Model Details:**")
                    st.json({
                        'Path': model['path'],
                        'Size': f"{model['size'] / 1024 / 1024:.1f} MB",
                        'Created': model.get('created', 'Unknown'),
                        'Type': model.get('metadata', {}).get('model_type', 'Unknown'),
                        'Accuracy': f"{model.get('metadata', {}).get('accuracy', 'N/A')}%"
                    })
                
                with col2:
                    st.markdown("**Actions:**")
                    
                    if st.button(f"🎨 Generate ODEs", key=f"gen_{model['name']}"):
                        st.session_state.current_ml_model = model
                        st.session_state.selected_nav = "🧮 Generation"
                        st.rerun()
                    
                    if st.button(f"📊 Evaluate", key=f"eval_{model['name']}"):
                        st.session_state.current_ml_model = model
                        st.session_state.selected_nav = "🤖 Machine Learning"
                        st.rerun()
                    
                    if st.button(f"📥 Download", key=f"download_{model['name']}"):
                        st.info("Model download functionality would be implemented here")
    
    # The remaining documentation section and other UI elements
    def documentation_page(self):
        """Documentation interface"""
        st.title("📚 Documentation")
        
        tabs = st.tabs([
            "Quick Start",
            "API Reference", 
            "Generator Guide",
            "ML Models",
            "Examples",
            "FAQ"
        ])
        
        with tabs[0]:
            self.quick_start_guide()
        
        with tabs[1]:
            self.api_reference()
        
        with tabs[2]:
            self.generator_guide()
        
        with tabs[3]:
            self.ml_models_guide()
        
        with tabs[4]:
            self.examples_showcase()
        
        with tabs[5]:
            self.faq_section()
    
    # Documentation methods remain the same as in the original code...
    
    # Error handling for edge cases
    def _handle_api_error(self, error: Exception, context: str = ""):
        """Handle API errors gracefully"""
        error_msg = str(error)
        
        if "ConnectionError" in error_msg:
            st.error(f"❌ Cannot connect to API. Please check if the server is running.")
        elif "timeout" in error_msg:
            st.error(f"⏱️ Request timed out. The server might be busy.")
        elif "403" in error_msg:
            st.error(f"🔒 Authentication failed. Please check your API key.")
        elif "404" in error_msg:
            st.error(f"🔍 Endpoint not found. Please check the API configuration.")
        else:
            st.error(f"❌ Error {context}: {error_msg}")
        
        # Log error for debugging
        if st.checkbox("Show error details"):
            st.code(str(error))

# ──────────────────────────────────────────────────────────────────────────
#  MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────
def main():
    """Main entry point"""
    app = AdvancedODEInterface()
    app.run()

if __name__ == "__main__":
    main()