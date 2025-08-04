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
import requests  # This was missing!

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
                st.experimental_rerun()
            return False
        
    def run(self):
        """Main interface runner"""
        # Check API connection
        api_available = self.check_api_connection()
        
        # Sidebar navigation
        st.sidebar.markdown("### Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["Generate ODEs", "ML Training", "AI Generation", "Analysis", "API Explorer", "Connection Test"]
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
        elif page == "API Explorer":
            self.api_explorer_page()
        elif page == "Connection Test":
            self.connection_test_page()
    
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
    
    def ml_training_page(self):
        """ML Model Training Interface"""
        st.title("🤖 ML Model Training")
        st.info("ML Training features coming soon!")
        
        # Placeholder for ML training interface
        st.markdown("""
        ### Available Features (Coming Soon):
        - Dataset selection and preprocessing
        - Model architecture selection
        - Training parameter configuration
        - Real-time training progress
        - Model evaluation and metrics
        """)
    
    def ai_generation_page(self):
        """AI-Powered ODE Generation"""
        st.title("🧠 AI-Powered ODE Generation")
        st.info("AI Generation features coming soon!")
        
        # Placeholder for AI generation
        st.markdown("""
        ### Available Features (Coming Soon):
        - Load pre-trained models
        - Generate novel ODEs using AI
        - Guided generation with constraints
        - Interactive ODE exploration
        """)
    
    def analysis_page(self):
        """Analysis Dashboard"""
        st.title("📊 ODE Analysis Dashboard")
        
        if not st.session_state.current_dataset:
            st.info("No ODEs in dataset. Generate some ODEs first!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.current_dataset)
        
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
                "POST /api/v1/verify"
            ]
        )
        
        # Show endpoint details and test
        if endpoint == "GET /health":
            st.markdown("### Health Check Endpoint")
            if st.button("Send Request"):
                url = API_BASE_URL.replace('/api/v1', '') + '/health'
                try:
                    response = requests.get(url, timeout=5)
                    st.code(f"GET {url}")
                    st.code(f"Status: {response.status_code}")
                    st.json(response.json())
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif endpoint == "POST /api/v1/generate":
            st.markdown("### Generate ODEs Endpoint")
            
            # Build request
            request_body = {
                "generator": st.selectbox("Generator", ["L1", "L2", "L3", "N1"]),
                "function": st.selectbox("Function", ["sine", "cosine", "exponential"]),
                "count": st.number_input("Count", 1, 10, 1),
                "verify": st.checkbox("Verify", True)
            }
            
            st.code(json.dumps(request_body, indent=2), language='json')
            
            if st.button("Send Request"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/generate",
                        headers=self.api_headers,
                        json=request_body,
                        timeout=10
                    )
                    
                    st.code(f"POST {API_BASE_URL}/generate")
                    st.code(f"Status: {response.status_code}")
                    st.json(response.json())
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Helper methods
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
    
    def _create_generator_chart(self, stats):
        """Create generator performance chart"""
        if not stats:
            return go.Figure()
            
        df = pd.DataFrame(stats)
        
        fig = go.Figure()
        
        if 'generator' in df.columns and 'count' in df.columns:
            fig.add_trace(go.Bar(
                x=df['generator'],
                y=df['count'],
                name='Count'
            ))
        
        fig.update_layout(
            title='Generator Performance',
            xaxis_title='Generator',
            yaxis_title='Count'
        )
        
        return fig
    
    def _get_system_metrics(self):
        """Get current system metrics"""
        try:
            response = requests.get(f"{API_BASE_URL}/stats", headers=self.api_headers, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        # Fallback metrics for demo
        return {
            'cpu_usage': np.random.uniform(20, 80),
            'cpu_delta': np.random.uniform(-5, 5),
            'memory_usage': np.random.uniform(30, 70),
            'memory_delta': np.random.uniform(-3, 3),
            'active_jobs': np.random.randint(0, 10),
            'generation_rate': np.random.uniform(10, 100),
            'history': []
        }

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
