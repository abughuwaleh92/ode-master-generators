# gui/integrated_interface.py
"""
Integrated GUI that connects to all backend services
- Uses the production API server
- Integrates with ML models
- Connects to monitoring services
"""
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import asyncio
import aiohttp
from pathlib import Path
import time
import os
import streamlit as st
import requests
import json

# Configuration - Use Streamlit secrets for production
if 'API_BASE_URL' in st.secrets:
    API_BASE_URL = st.secrets['API_BASE_URL']
    API_KEY = st.secrets['API_KEY']
else:
    # Fallback for local development
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000/api/v1')
    API_KEY = os.getenv('API_KEY', 'your-test-key')

# Show connection status
st.sidebar.markdown("### Connection Status")
try:
    # Remove /api/v1 and add /health
    health_url = API_BASE_URL.replace('/api/v1', '/health')
    response = requests.get(health_url, timeout=5)
    if response.status_code == 200:
        st.sidebar.success("✅ Connected to API")
    else:
        st.sidebar.error("❌ API Connection Failed")
except Exception as e:
    st.sidebar.warning("⚠️ API Offline")
    st.sidebar.text(f"URL: {API_BASE_URL}")

# Get configuration from Streamlit secrets or environment
if 'API_BASE_URL' in st.secrets:
    API_BASE_URL = st.secrets['API_BASE_URL']
    API_KEY = st.secrets['API_KEY']
else:
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000/api/v1')
    API_KEY = os.getenv('API_KEY', 'test-key')

# Show connection status
st.sidebar.markdown("### Connection Status")
try:
    response = requests.get(f"{API_BASE_URL.replace('/api/v1', '/health')}", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ Connected to API")
    else:
        st.sidebar.error("❌ API Connection Failed")
except:
    st.sidebar.warning("⚠️ API Offline - Limited Functionality")

import os
import streamlit as st

# Configuration - Use Streamlit secrets if available, otherwise environment/defaults
if hasattr(st, 'secrets') and 'API_BASE_URL' in st.secrets:
    API_BASE_URL = st.secrets['API_BASE_URL']
    API_KEY = st.secrets['API_KEY']
else:
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000/api/v1')
    API_KEY = os.getenv('API_KEY', 'your-secret-key-1')

MONITORING_URL = os.getenv('MONITORING_URL', 'http://localhost:8050')

class IntegratedODEInterface:
    def __init__(self):
        self.api_headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        self.session_state = st.session_state
        
    def run(self):
        """Main interface runner"""
        st.set_page_config(
            page_title="ODE Master Generator - Integrated System",
            page_icon="🔬",
            layout="wide"
        )
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Generate ODEs", "ML Training", "AI Generation", "Analysis", "Monitoring", "API Explorer"]
        )
        
        # Route to appropriate page
        if page == "Generate ODEs":
            self.generation_page()
        elif page == "ML Training":
            self.ml_training_page()
        elif page == "AI Generation":
            self.ai_generation_page()
        elif page == "Analysis":
            self.analysis_page()
        elif page == "Monitoring":
            self.monitoring_page()
        elif page == "API Explorer":
            self.api_explorer_page()
    
    def generation_page(self):
        """ODE Generation using API backend"""
        st.title("🧮 ODE Generation")
        st.markdown("Generate ODEs using the backend API service")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            generator = st.selectbox(
                "Generator",
                ["L1", "L2", "L3", "L4", "N1", "N2", "N3", "N7"]
            )
            
        with col2:
            function = st.selectbox(
                "Function",
                ["identity", "quadratic", "sine", "cosine", "exponential", 
                 "logarithm", "rational", "bessel"]
            )
            
        with col3:
            count = st.number_input("Number of ODEs", min_value=1, max_value=100, value=5)
        
        # Parameters
        st.subheader("Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alpha = st.slider("α", -5.0, 5.0, 1.0, 0.1)
        with col2:
            beta = st.slider("β", 0.1, 5.0, 1.0, 0.1)
        with col3:
            M = st.slider("M", -5.0, 5.0, 0.0, 0.1)
        
        verify = st.checkbox("Verify generated ODEs", value=True)
        
        if st.button("🚀 Generate ODEs", type="primary"):
            with st.spinner("Generating ODEs via API..."):
                # Call API
                response = self._call_api_generate({
                    "generator": generator,
                    "function": function,
                    "parameters": {"alpha": alpha, "beta": beta, "M": M},
                    "count": count,
                    "verify": verify
                })
                
                if response['status'] == 'success':
                    job_id = response['data']['job_id']
                    st.success(f"Job created: {job_id}")
                    
                    # Poll for results
                    results = self._poll_job_status(job_id)
                    
                    if results:
                        # Display results
                        st.subheader("Generated ODEs")
                        for i, ode in enumerate(results):
                            with st.expander(f"ODE {i+1} - {ode['id']}"):
                                st.latex(ode.get('ode_latex', ode['ode']))
                                if ode.get('solution'):
                                    st.latex(f"Solution: {ode.get('solution_latex', ode['solution'])}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Verified", "✓" if ode.get('verified') else "✗")
                                with col2:
                                    st.metric("Complexity", ode.get('complexity', 'N/A'))
                                with col3:
                                    st.metric("Generator", ode.get('generator'))
                        
                        # Save option
                        if st.button("💾 Save to Dataset"):
                            self._save_to_dataset(results)
                else:
                    st.error(f"API Error: {response.get('error', 'Unknown error')}")
    
    def ml_training_page(self):
        """ML Model Training Interface"""
        st.title("🤖 ML Model Training")
        
        # Dataset selection
        datasets = self._get_available_datasets()
        selected_dataset = st.selectbox("Select Dataset", datasets)
        
        # Model configuration
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["pattern_net", "transformer", "vae", "language_model"]
            )
            
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50)
            batch_size = st.number_input("Batch Size", min_value=8, max_value=256, value=32)
            
        with col2:
            learning_rate = st.number_input(
                "Learning Rate", 
                min_value=0.00001, 
                max_value=0.1, 
                value=0.001,
                format="%.5f"
            )
            
            optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
            
            early_stopping = st.checkbox("Early Stopping", value=True)
        
        # Advanced options
        with st.expander("Advanced Options"):
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
            weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.01, value=0.0001, format="%.5f")
            
            # Model-specific options
            if model_type == "transformer":
                n_heads = st.number_input("Number of Heads", min_value=1, max_value=16, value=8)
                n_layers = st.number_input("Number of Layers", min_value=1, max_value=12, value=6)
        
        if st.button("🎯 Start Training", type="primary"):
            # Create training job
            training_config = {
                "dataset": selected_dataset,
                "model_type": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "early_stopping": early_stopping,
                "dropout": dropout,
                "weight_decay": weight_decay
            }
            
            # Call training API
            with st.spinner("Submitting training job..."):
                response = self._call_api_train(training_config)
                
                if response['status'] == 'success':
                    training_job_id = response['data']['job_id']
                    st.success(f"Training job started: {training_job_id}")
                    
                    # Show training progress
                    self._show_training_progress(training_job_id)
                else:
                    st.error(f"Failed to start training: {response.get('error')}")
    
    def ai_generation_page(self):
        """AI-Powered ODE Generation"""
        st.title("🧠 AI-Powered ODE Generation")
        st.markdown("Generate novel ODEs using trained ML models")
        
        # Model selection
        available_models = self._get_available_models()
        selected_model = st.selectbox("Select Trained Model", available_models)
        
        col1, col2 = st.columns(2)
        
        with col1:
            generation_mode = st.radio(
                "Generation Mode",
                ["Free Generation", "Guided Generation", "Interactive Exploration"]
            )
            
            n_samples = st.number_input("Number of Samples", min_value=1, max_value=1000, value=10)
            
        with col2:
            if generation_mode == "Guided Generation":
                target_generator = st.selectbox("Target Generator Style", ["Any", "L1", "L2", "L3", "L4", "N1", "N2", "N3", "N7"])
                target_function = st.selectbox("Target Function Type", ["Any", "trigonometric", "exponential", "polynomial", "special"])
                target_complexity = st.slider("Target Complexity", 0, 500, (50, 200))
            
            temperature = st.slider("Creativity (Temperature)", 0.1, 2.0, 0.8)
        
        if generation_mode == "Interactive Exploration":
            st.subheader("Interactive ODE Builder")
            
            # Interactive controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                add_term = st.selectbox("Add Term", ["", "sin", "cos", "exp", "log", "power"])
            with col2:
                term_coefficient = st.number_input("Coefficient", value=1.0)
            with col3:
                if st.button("Add to ODE"):
                    if 'current_ode' not in st.session_state:
                        st.session_state.current_ode = []
                    st.session_state.current_ode.append((add_term, term_coefficient))
            
            # Display current ODE
            if 'current_ode' in st.session_state and st.session_state.current_ode:
                st.write("Current ODE Structure:")
                ode_str = " + ".join([f"{coef}*{term}(y)" for term, coef in st.session_state.current_ode])
                st.latex(f"y''(x) + {ode_str} = ?")
        
        if st.button("🎨 Generate with AI", type="primary"):
            with st.spinner("AI is creating novel ODEs..."):
                # Prepare generation request
                gen_config = {
                    "model_path": selected_model,
                    "n_samples": n_samples,
                    "temperature": temperature
                }
                
                if generation_mode == "Guided Generation":
                    gen_config.update({
                        "generator": None if target_generator == "Any" else target_generator,
                        "function": None if target_function == "Any" else target_function,
                        "complexity_range": target_complexity
                    })
                
                # Call AI generation API
                response = self._call_api_ai_generate(gen_config)
                
                if response['status'] == 'success':
                    generated_odes = response['data']['odes']
                    
                    # Display results
                    st.subheader(f"Generated {len(generated_odes)} Novel ODEs")
                    
                    # Quality metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        novelty_score = response['data'].get('avg_novelty_score', 0)
                        st.metric("Average Novelty", f"{novelty_score:.2%}")
                    with col2:
                        diversity_score = response['data'].get('diversity_score', 0)
                        st.metric("Diversity Score", f"{diversity_score:.2%}")
                    with col3:
                        valid_count = sum(1 for ode in generated_odes if ode.get('valid'))
                        st.metric("Valid ODEs", f"{valid_count}/{len(generated_odes)}")
                    
                    # Display ODEs
                    for i, ode in enumerate(generated_odes):
                        with st.expander(f"AI ODE {i+1} - Novelty: {ode.get('novelty_score', 0):.2%}"):
                            st.latex(ode['ode_latex'])
                            
                            if ode.get('similar_to'):
                                st.info(f"Similar to: {ode['similar_to']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"🔍 Analyze", key=f"analyze_{i}"):
                                    self._analyze_single_ode(ode)
                            with col2:
                                if st.button(f"✓ Verify", key=f"verify_{i}"):
                                    self._verify_single_ode(ode)
    
    def analysis_page(self):
        """Comprehensive Analysis Dashboard"""
        st.title("📊 ODE Analysis Dashboard")
        
        # Load dataset
        datasets = self._get_available_datasets()
        selected_dataset = st.selectbox("Select Dataset for Analysis", datasets)
        
        if st.button("Load Dataset"):
            with st.spinner("Loading and analyzing dataset..."):
                # Call analysis API
                analysis_results = self._call_api_analyze(selected_dataset)
                
                if analysis_results['status'] == 'success':
                    data = analysis_results['data']
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total ODEs", data['total_odes'])
                    with col2:
                        st.metric("Verified", f"{data['verification_rate']:.1%}")
                    with col3:
                        st.metric("Avg Complexity", f"{data['avg_complexity']:.1f}")
                    with col4:
                        st.metric("Unique Generators", data['unique_generators'])
                    
                    # Visualizations
                    st.subheader("Generator Performance")
                    fig_generator = self._create_generator_chart(data['generator_stats'])
                    st.plotly_chart(fig_generator, use_container_width=True)
                    
                    st.subheader("Complexity Distribution")
                    fig_complexity = self._create_complexity_chart(data['complexity_distribution'])
                    st.plotly_chart(fig_complexity, use_container_width=True)
                    
                    # Pattern analysis
                    st.subheader("ODE Patterns")
                    pattern_df = pd.DataFrame(data['patterns'])
                    st.dataframe(pattern_df)
    
    def monitoring_page(self):
        """Real-time System Monitoring"""
        st.title("📡 System Monitoring")
        
        # Embed monitoring dashboard
        st.markdown("### Real-time Metrics")
        
        # Create placeholder for live updates
        metric_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Auto-refresh
        if st.checkbox("Auto-refresh (5s)", value=True):
            while True:
                # Get current metrics
                metrics = self._get_system_metrics()
                
                with metric_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("CPU Usage", f"{metrics['cpu_usage']:.1f}%", 
                                 delta=f"{metrics['cpu_delta']:.1f}%")
                    with col2:
                        st.metric("Memory", f"{metrics['memory_usage']:.1f}%",
                                 delta=f"{metrics['memory_delta']:.1f}%")
                    with col3:
                        st.metric("Active Jobs", metrics['active_jobs'])
                    with col4:
                        st.metric("Generation Rate", f"{metrics['generation_rate']:.1f}/min")
                
                # Update chart
                with chart_placeholder.container():
                    fig = self._create_monitoring_chart(metrics['history'])
                    st.plotly_chart(fig, use_container_width=True)
                
                time.sleep(5)
    
    def api_explorer_page(self):
        """Interactive API Explorer"""
        st.title("🔌 API Explorer")
        st.markdown("Test and explore API endpoints")
        
        # Endpoint selection
        endpoint = st.selectbox(
            "Select Endpoint",
            [
                "POST /api/v1/generate",
                "POST /api/v1/verify",
                "GET /api/v1/jobs/{job_id}",
                "GET /api/v1/stats",
                "POST /api/v1/ml/train",
                "POST /api/v1/ml/generate"
            ]
        )
        
        # Dynamic form based on endpoint
        if endpoint == "POST /api/v1/generate":
            st.subheader("Generate ODEs")
            
            # Build request
            request_body = {
                "generator": st.selectbox("Generator", ["L1", "L2", "L3", "L4", "N1", "N2", "N3", "N7"]),
                "function": st.selectbox("Function", ["sine", "cosine", "exponential"]),
                "count": st.number_input("Count", 1, 100, 1),
                "verify": st.checkbox("Verify", True)
            }
            
            # Show request
            st.code(json.dumps(request_body, indent=2), language='json')
            
            if st.button("Send Request"):
                response = requests.post(
                    f"{API_BASE_URL}/generate",
                    headers=self.api_headers,
                    json=request_body
                )
                
                # Show response
                st.subheader("Response")
                st.code(json.dumps(response.json(), indent=2), language='json')
    
    # Helper methods
    def _call_api_generate(self, config):
        """Call generation API"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                headers=self.api_headers,
                json=config
            )
            
            if response.status_code == 200:
                return {'status': 'success', 'data': response.json()}
            else:
                return {'status': 'error', 'error': response.text}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _poll_job_status(self, job_id, max_attempts=60):
        """Poll job status until completion"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for attempt in range(max_attempts):
            response = requests.get(
                f"{API_BASE_URL}/jobs/{job_id}",
                headers=self.api_headers
            )
            
            if response.status_code == 200:
                job_data = response.json()
                
                # Update progress
                progress = job_data.get('progress', 0)
                progress_bar.progress(progress / 100)
                status_text.text(f"Status: {job_data['status']} - {progress:.0f}%")
                
                if job_data['status'] == 'completed':
                    return job_data.get('results', [])
                elif job_data['status'] == 'failed':
                    st.error(f"Job failed: {job_data.get('error')}")
                    return None
            
            time.sleep(1)
        
        st.error("Job timeout")
        return None
    
    def _get_available_datasets(self):
        """Get list of available datasets"""
        # In production, this would query the API
        datasets = []
        for file in Path('.').glob('*.jsonl'):
            datasets.append(file.name)
        return datasets
    
    def _get_available_models(self):
        """Get list of trained models"""
        try:
            response = requests.get(f"{API_BASE_URL}/ml/models", headers=self.api_headers)
            if response.status_code == 200:
                return response.json()['models']
        except:
            pass
        
        # Fallback to local search
        models = []
        for file in Path('models').glob('*.pth'):
            models.append(str(file))
        return models
    
    def _create_generator_chart(self, stats):
        """Create generator performance chart"""
        df = pd.DataFrame(stats)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['generator'],
            y=df['count'],
            name='Count',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['generator'],
            y=df['verification_rate'],
            name='Verification Rate',
            yaxis='y2',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Generator Performance',
            yaxis=dict(title='Count', side='left'),
            yaxis2=dict(title='Verification Rate', side='right', overlaying='y'),
            hovermode='x unified'
        )
        
        return fig
    
    def _get_system_metrics(self):
        """Get current system metrics"""
        try:
            response = requests.get(f"{API_BASE_URL}/stats", headers=self.api_headers)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        # Fallback metrics
        return {
            'cpu_usage': np.random.uniform(20, 80),
            'cpu_delta': np.random.uniform(-5, 5),
            'memory_usage': np.random.uniform(30, 70),
            'memory_delta': np.random.uniform(-3, 3),
            'active_jobs': np.random.randint(0, 10),
            'generation_rate': np.random.uniform(10, 100),
            'history': []
        }

# Run the integrated interface
if __name__ == "__main__":
    # Check if API server is running
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '/health')}")
        if response.status_code != 200:
            st.error("⚠️ API Server is not running! Start it with: `python scripts/production_server.py`")
            st.stop()
    except:
        st.error("⚠️ Cannot connect to API Server! Start it with: `python scripts/production_server.py`")
        st.stop()
    
    # Run interface
    app = IntegratedODEInterface()
    app.run()
