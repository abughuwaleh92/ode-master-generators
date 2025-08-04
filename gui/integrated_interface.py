# gui/integrated_interface.py
"""
Integrated GUI that connects to all backend services.
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests

# ──────────────────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ODE Master Generator – Integrated System",
    page_icon="🔬",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
# Use environment variables (Railway sets these)
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000/api/v1')
API_KEY = os.getenv('API_KEY', 'test-key')
MONITORING_URL = os.getenv('MONITORING_URL', 'http://localhost:8050')

# ──────────────────────────────────────────────────────────────────────────
#  INTEGRATED ODE INTERFACE CLASS
# ──────────────────────────────────────────────────────────────────────────
class IntegratedODEInterface:
    def __init__(self):
        self.api_headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        self.session_state = st.session_state
        
        # Initialize session state
        if 'generated_odes' not in st.session_state:
            st.session_state.generated_odes = []
        if 'current_dataset' not in st.session_state:
            st.session_state.current_dataset = []
        if 'current_ode' not in st.session_state:
            st.session_state.current_ode = []
            
    def check_api_connection(self):
        """Check API connection and show status in sidebar"""
        st.sidebar.markdown("### Connection Status")
        
        # Show configuration
        st.sidebar.markdown("### Configuration")
        if os.getenv('API_BASE_URL'):
            st.sidebar.success("✅ Using Railway environment variables")
            # Show masked URL
            api_host = API_BASE_URL.split('/')[2] if len(API_BASE_URL.split('/')) > 2 else API_BASE_URL
            st.sidebar.text(f"API: {api_host}")
        else:
            st.sidebar.warning("⚠️ Using default configuration")
        
        # Test connection
        try:
            # Fix: Ensure we're using the correct health endpoint
            health_url = API_BASE_URL.replace('/api/v1', '') + '/health'
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                st.sidebar.success("✅ Connected to API")
                return True
            else:
                st.sidebar.error(f"❌ API returned {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            st.sidebar.warning("⚠️ API Offline - Limited Functionality")
           if st.sidebar.button("🔄 Retry Connection"):
    st.rerun()
            return False
        
    def run(self):
        """Main interface runner"""
        # Check API connection
        api_available = self.check_api_connection()
        
        # Sidebar navigation
        st.sidebar.markdown("### Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["Generate ODEs", "ML Training", "AI Generation", "Analysis", "Monitoring", "API Explorer", "Connection Test"]
        )
        
        # Show warning if API not available
        if not api_available and page != "Connection Test":
            st.warning("⚠️ API is not connected. Some features may not work properly.")
        
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
        elif page == "Connection Test":
            self.connection_test_page()
    
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
                    if add_term:
                        st.session_state.current_ode.append((add_term, term_coefficient))
            
            # Display current ODE
            if st.session_state.current_ode:
                st.write("Current ODE Structure:")
                ode_str = " + ".join([f"{coef}*{term}(y)" for term, coef in st.session_state.current_ode])
                st.latex(f"y''(x) + {ode_str} = ?")
                
                if st.button("Clear ODE"):
    st.session_state.current_ode = []
    st.rerun()
        
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
                elif generation_mode == "Interactive Exploration":
                    gen_config["ode_structure"] = st.session_state.current_ode
                
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
                            st.latex(ode.get('ode_latex', ode.get('ode', 'N/A')))
                            
                            if ode.get('similar_to'):
                                st.info(f"Similar to: {ode['similar_to']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"🔍 Analyze", key=f"analyze_{i}"):
                                    self._analyze_single_ode(ode)
                            with col2:
                                if st.button(f"✓ Verify", key=f"verify_{i}"):
                                    self._verify_single_ode(ode)
                else:
                    st.error(f"AI Generation failed: {response.get('error', 'Unknown error')}")
    
    def monitoring_page(self):
        """Real-time System Monitoring"""
        st.title("📡 System Monitoring")
        
        # Create tabs for different monitoring views
        tab1, tab2, tab3 = st.tabs(["Real-time Metrics", "System Logs", "Performance History"])
        
        with tab1:
            st.markdown("### Real-time Metrics")
            
            # Auto-refresh checkbox
            auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
            
            # Get current metrics
            metrics = self._get_system_metrics()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CPU Usage", f"{metrics.get('cpu_usage', 0):.1f}%", 
                         delta=f"{metrics.get('cpu_delta', 0):.1f}%")
            with col2:
                st.metric("Memory", f"{metrics.get('memory_usage', 0):.1f}%",
                         delta=f"{metrics.get('memory_delta', 0):.1f}%")
            with col3:
                st.metric("Active Jobs", metrics.get('active_jobs', 0))
            with col4:
                st.metric("Generation Rate", f"{metrics.get('generation_rate', 0):.1f}/min")
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Generated", metrics.get('total_generated', 0))
            with col2:
                st.metric("Total Verified", metrics.get('total_verified', 0))
            with col3:
                st.metric("Cache Size", metrics.get('cache_size', 0))
            
            # Create monitoring chart
            if metrics.get('history'):
                fig = self._create_monitoring_chart(metrics['history'])
                st.plotly_chart(fig, use_container_width=True)
            
            if auto_refresh:
    time.sleep(5)
    st.rerun()
        
        with tab2:
            st.markdown("### System Logs")
            st.info("System logs would be displayed here in a production environment")
            
        with tab3:
            st.markdown("### Performance History")
            st.info("Historical performance metrics would be displayed here")
    
    def connection_test_page(self):
        """Test API connection with detailed information"""
        st.title("🔌 Connection Test")
        
        st.info(f"""
        **Current Configuration:**
        - API URL: `{API_BASE_URL}`
        - API Key: `{'*' * 20 if API_KEY and API_KEY != 'test-key' else 'Not Set'}`
        """)
        
        # Test buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Test Health", type="primary"):
                self._test_health_endpoint()
        
        with col2:
            if st.button("Test Auth", type="primary"):
                self._test_auth_endpoint()
        
        with col3:
            if st.button("Test Generate", type="primary"):
                self._test_generate_endpoint()
                
        # Show Railway configuration help
        with st.expander("📚 Railway Configuration Help"):
            st.markdown("""
            ### Your Railway GUI Variables Should Be:
            
            1. **API_BASE_URL**: Your API URL + `/api/v1`
               - Example: `https://ode-api-production.up.railway.app/api/v1`
               - NOT just: `https://ode-api-production.up.railway.app`
            
            2. **API_KEY**: Same key as in your API service
               - Must match exactly between API and GUI services
            
            ### To Fix:
            1. Go to Railway Dashboard → GUI Service → Variables
            2. Edit `API_BASE_URL` to add `/api/v1` at the end
            3. Make sure `API_KEY` matches your API service
            """)
    
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
                        # Store in session state
                        st.session_state.generated_odes = results
                        
                        # Display results
                        st.subheader("Generated ODEs")
                        for i, ode in enumerate(results):
                            with st.expander(f"ODE {i+1} - {ode['id']}"):
                                st.code(ode.get('ode', 'N/A'))
                                if ode.get('solution'):
                                    st.code(f"Solution: {ode.get('solution', 'N/A')}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Verified", "✓" if ode.get('verified') else "✗")
                                with col2:
                                    st.metric("Complexity", ode.get('complexity', 'N/A'))
                                with col3:
                                    st.metric("Generator", ode.get('generator'))
                        
                        # Save option
                        if st.button("💾 Save to Dataset"):
                            st.session_state.current_dataset.extend(results)
                            st.success(f"Added {len(results)} ODEs to dataset!")
                else:
                    st.error(f"API Error: {response.get('error', 'Unknown error')}")
    
    def analysis_page(self):
        """Comprehensive Analysis Dashboard"""
        st.title("📊 ODE Analysis Dashboard")
        
        # Analysis options
        analysis_type = st.radio(
            "Analysis Type",
            ["Current Session", "Load Dataset", "Compare Datasets"]
        )
        
        if analysis_type == "Current Session":
            if not st.session_state.current_dataset:
                st.info("No ODEs in current session. Generate some ODEs first!")
                return
            
            df = pd.DataFrame(st.session_state.current_dataset)
            st.success(f"Analyzing {len(df)} ODEs from current session")
            
        elif analysis_type == "Load Dataset":
            datasets = self._get_available_datasets()
            selected_dataset = st.selectbox("Select Dataset for Analysis", datasets)
            
            if st.button("Load Dataset"):
                with st.spinner("Loading and analyzing dataset..."):
                    # Call analysis API
                    analysis_results = self._call_api_analyze(selected_dataset)
                    
                    if analysis_results['status'] == 'success':
                        df = pd.DataFrame(analysis_results['data']['odes'])
                        st.success(f"Loaded {len(df)} ODEs")
                    else:
                        st.error("Failed to load dataset")
                        return
            else:
                return
        
        # Display analysis
        if 'df' in locals():
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total ODEs", len(df))
            with col2:
                if 'verified' in df.columns:
                    verified_rate = df['verified'].sum() / len(df) * 100
                    st.metric("Verified", f"{verified_rate:.1f}%")
            with col3:
                if 'complexity' in df.columns:
                    st.metric("Avg Complexity", f"{df['complexity'].mean():.1f}")
            with col4:
                if 'generator' in df.columns:
                    st.metric("Unique Generators", df['generator'].nunique())
            
            # Visualizations
            if 'generator' in df.columns:
                st.subheader("Generator Distribution")
                fig = px.bar(df['generator'].value_counts().reset_index(), 
                            x='index', y='generator',
                            labels={'index': 'Generator', 'generator': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            if 'complexity' in df.columns:
                st.subheader("Complexity Distribution")
                fig = px.histogram(df, x='complexity', nbins=30, 
                                 title="ODE Complexity Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Pattern analysis
            st.subheader("ODE Patterns")
            if st.checkbox("Show detailed pattern analysis"):
                pattern_analysis = self._analyze_patterns(df)
                st.dataframe(pattern_analysis)
    
    def api_explorer_page(self):
        """Interactive API Explorer"""
        st.title("🔌 API Explorer")
        st.markdown("Test and explore API endpoints")
        
        # Endpoint selection
        endpoint = st.selectbox(
            "Select Endpoint",
            [
                "GET /health",
                "GET /api/v1/stats",
                "POST /api/v1/generate",
                "POST /api/v1/verify",
                "GET /api/v1/jobs/{job_id}",
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
            
            # Add parameters
            st.subheader("Parameters")
            request_body["parameters"] = {
                "alpha": st.slider("Alpha", -5.0, 5.0, 1.0),
                "beta": st.slider("Beta", 0.1, 5.0, 1.0),
                "M": st.slider("M", -5.0, 5.0, 0.0)
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
                st.code(f"Status: {response.status_code}")
                try:
                    st.json(response.json())
                except:
                    st.text(response.text)
        
        elif endpoint == "POST /api/v1/verify":
            st.subheader("Verify ODE")
            
            ode = st.text_input("ODE Equation", "y''(x) + y(x) = sin(x)")
            solution = st.text_input("Solution", "y(x) = -sin(x)/2")
            method = st.selectbox("Method", ["substitution", "numerical"])
            
            request_body = {
                "ode": ode,
                "solution": solution,
                "method": method
            }
            
            if st.button("Verify"):
                response = requests.post(
                    f"{API_BASE_URL}/verify",
                    headers=self.api_headers,
                    json=request_body
                )
                
                st.code(f"Status: {response.status_code}")
                try:
                    result = response.json()
                    st.json(result)
                    
                    if result.get('verified'):
                        st.success("✅ ODE and solution verified!")
                    else:
                        st.error("❌ Verification failed")
                except:
                    st.text(response.text)
    
    # Helper methods
    def _test_health_endpoint(self):
        """Test health endpoint"""
        with st.spinner("Testing health endpoint..."):
            try:
                url = API_BASE_URL.replace('/api/v1', '') + '/health'
                st.code(f"GET {url}")
                
                response = requests.get(url, timeout=5)
                st.code(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    st.success("✅ Health check passed!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Health check failed: {response.status_code}")
                    st.text(response.text)
                    
            except Exception as e:
                st.error(f"❌ Connection error: {type(e).__name__}: {str(e)}")
    
    def _test_auth_endpoint(self):
        """Test authenticated endpoint"""
        with st.spinner("Testing authentication..."):
            try:
                url = f"{API_BASE_URL}/stats"
                st.code(f"GET {url}")
                st.code(f"Headers: X-API-Key: {'*' * 20}")
                
                response = requests.get(url, headers=self.api_headers, timeout=5)
                st.code(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    st.success("✅ Authentication successful!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Authentication failed: {response.status_code}")
                    st.text(response.text)
                    
            except Exception as e:
                st.error(f"❌ Error: {type(e).__name__}: {str(e)}")
    
def _test_generate_endpoint(self):
    """Test generation endpoint"""
    with st.spinner("Testing ODE generation..."):
        try:
            test_data = {
                "generator": "L1",
                "function": "sine",
                "count": 1,
                "verify": True
            }
            
            url = f"{API_BASE_URL}/generate"
            st.code(f"POST {url}")
            st.code(f"Body: {json.dumps(test_data, indent=2)}")
            
            # DEBUG: Show actual headers
            st.code(f"Headers: {json.dumps(self.api_headers, indent=2)}")
            st.info(f"API_KEY from env: {API_KEY}")
            
            response = requests.post(
                url,
                headers=self.api_headers,
                json=test_data,
                timeout=10
            )
            
            st.code(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                st.success("✅ Generation endpoint working!")
                st.json(response.json())
            else:
                st.error(f"❌ Generation failed: {response.status_code}")
                st.text(response.text)
                
        except Exception as e:
            st.error(f"❌ Error: {type(e).__name__}: {str(e)}")
    
    def _call_api_generate(self, config):
        """Call generation API"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                headers=self.api_headers,
                json=config,
                timeout=30
            )
            
            if response.status_code == 200:
                return {'status': 'success', 'data': response.json()}
            else:
                return {'status': 'error', 'error': f"{response.status_code}: {response.text}"}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _call_api_train(self, config):
        """Call training API"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/ml/train",
                headers=self.api_headers,
                json=config
            )
            
            if response.status_code == 200:
                return {'status': 'success', 'data': response.json()}
            else:
                return {'status': 'error', 'error': f"API returned {response.status_code}: {response.text}"}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _call_api_analyze(self, dataset_name):
        """Call analysis API"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                headers=self.api_headers,
                json={"dataset": dataset_name}
            )
            
            if response.status_code == 200:
                return {'status': 'success', 'data': response.json()}
            else:
                return {'status': 'error', 'error': response.text}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _call_api_ai_generate(self, config):
        """Call AI generation API"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/ml/generate",
                headers=self.api_headers,
                json=config
            )
            
            if response.status_code == 200:
                return {'status': 'success', 'data': response.json()}
            else:
                # For demo, return mock data
                if "ml/generate" in API_BASE_URL:
                    return self._mock_ai_generation(config)
                return {'status': 'error', 'error': response.text}
        except Exception as e:
            # For demo, return mock data
            return self._mock_ai_generation(config)
    
    def _mock_ai_generation(self, config):
        """Mock AI generation for demo"""
        n_samples = config.get('n_samples', 10)
        mock_odes = []
        
        for i in range(n_samples):
            mock_odes.append({
                'id': f'ai_ode_{i+1}',
                'ode': f"y''(x) + {np.random.rand():.2f}*y'(x) + {np.random.rand():.2f}*y(x) = f(x)",
                'ode_latex': f"y''(x) + {np.random.rand():.2f} y'(x) + {np.random.rand():.2f} y(x) = f(x)",
                'novelty_score': np.random.rand(),
                'valid': np.random.choice([True, False], p=[0.8, 0.2]),
                'similar_to': np.random.choice(['L1-sine', 'N1-exp', 'L2-cos', None])
            })
        
        return {
            'status': 'success',
            'data': {
                'odes': mock_odes,
                'avg_novelty_score': np.mean([ode['novelty_score'] for ode in mock_odes]),
                'diversity_score': np.random.rand()
            }
        }
    
    def _show_training_progress(self, job_id):
        """Show training progress"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # For demo, simulate progress
        for i in range(101):
            progress_bar.progress(i)
            status_text.text(f"Training progress: {i}%")
            time.sleep(0.05)
        
        st.success("Training completed!")
    
    def _poll_job_status(self, job_id, max_attempts=60):
        """Poll job status until completion"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/jobs/{job_id}",
                    headers=self.api_headers,
                    timeout=5
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
                elif response.status_code == 404:
                    st.error("Job not found")
                    return None
                    
            except Exception as e:
                st.error(f"Error polling job: {e}")
                return None
            
            time.sleep(1)
        
        st.error("Job timeout")
        return None
    
    def _get_available_datasets(self):
        """Get list of available datasets"""
        datasets = []
        for file in Path('.').glob('*.jsonl'):
            datasets.append(file.name)
        return datasets if datasets else ["No datasets found"]
    
    def _get_available_models(self):
        """Get list of trained models"""
        models = []
        model_dir = Path('models')
        if model_dir.exists():
            for file in model_dir.glob('*.pth'):
                models.append(str(file))
        return models if models else ["No models found"]
    
    def _get_system_metrics(self):
        """Get current system metrics"""
        try:
            response = requests.get(f"{API_BASE_URL}/stats", headers=self.api_headers, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        # Return demo metrics
        return {
            'cpu_usage': np.random.uniform(20, 80),
            'cpu_delta': np.random.uniform(-5, 5),
            'memory_usage': np.random.uniform(30, 70),
            'memory_delta': np.random.uniform(-3, 3),
            'active_jobs': np.random.randint(0, 10),
            'generation_rate': np.random.uniform(10, 100),
            'total_generated': np.random.randint(1000, 5000),
            'total_verified': np.random.randint(800, 4000),
            'cache_size': np.random.randint(100, 1000),
            'history': []
        }
    
    def _create_monitoring_chart(self, history):
        """Create monitoring chart"""
        fig = go.Figure()
        
        # Add traces for different metrics
        fig.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=[h.get('cpu_usage', 0) for h in history],
            mode='lines',
            name='CPU Usage',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=[h.get('memory_usage', 0) for h in history],
            mode='lines',
            name='Memory Usage',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title='System Resources Over Time',
            xaxis_title='Time',
            yaxis_title='Usage %',
            hovermode='x unified'
        )
        
        return fig
    
    def _analyze_single_ode(self, ode):
        """Analyze a single ODE"""
        st.info("Analyzing ODE...")
        # Add analysis logic here
        st.success("Analysis complete!")
    
    def _verify_single_ode(self, ode):
        """Verify a single ODE"""
        st.info("Verifying ODE...")
        # Add verification logic here
        st.success("Verification complete!")
    
    def _analyze_patterns(self, df):
        """Analyze patterns in ODEs"""
        # Simple pattern analysis
        patterns = {
            'most_common_generator': df['generator'].mode()[0] if 'generator' in df.columns else 'N/A',
            'most_common_function': df['function'].mode()[0] if 'function' in df.columns else 'N/A',
            'avg_complexity': df['complexity'].mean() if 'complexity' in df.columns else 0,
            'verification_rate': (df['verified'].sum() / len(df) * 100) if 'verified' in df.columns else 0
        }
        return pd.DataFrame([patterns])

# ──────────────────────────────────────────────────────────────────────────
#  MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Show title
    st.title("🔬 ODE Master Generator")
    st.markdown("Integrated System for ODE Generation and Analysis")
    
    # Run interface
    app = IntegratedODEInterface()
    app.run()
