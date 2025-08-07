import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
import time
import base64
from io import BytesIO
import sympy as sp
from sympy import symbols, Function, dsolve, Eq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
import os

# Page configuration
st.set_page_config(
    page_title="ODE Master Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .ode-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 5px;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'https://your-api.railway.app')
API_KEY = os.getenv('API_KEY', 'your-api-key')

# Initialize session state
if 'generated_odes' not in st.session_state:
    st.session_state.generated_odes = []
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

class ODEAPIClient:
    """API client for ODE generation system"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
    
    def generate_odes(self, generator: str, function: str, parameters: Dict, count: int = 1, verify: bool = True) -> Dict:
        """Generate ODEs using the API"""
        payload = {
            'generator': generator,
            'function': function,
            'parameters': parameters,
            'count': count,
            'verify': verify
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/generate",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get job status"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/jobs/{job_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def verify_ode(self, ode: str, solution: str) -> Dict:
        """Verify ODE solution"""
        payload = {
            'ode': ode,
            'solution': solution,
            'method': 'substitution'
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/verify",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def get_generators(self) -> Dict:
        """Get available generators"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/generators",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return {'all': []}
    
    def get_functions(self) -> Dict:
        """Get available functions"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/functions",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return {'functions': []}
    
    def get_statistics(self) -> Dict:
        """Get API statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/stats",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    def analyze_dataset(self, ode_list: List[str]) -> Dict:
        """Analyze ODE dataset"""
        payload = {
            'ode_list': ode_list,
            'analysis_type': 'comprehensive'
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/analyze",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def train_model(self, dataset: str, model_type: str, epochs: int, config: Dict) -> Dict:
        """Train ML model"""
        payload = {
            'dataset': dataset,
            'model_type': model_type,
            'epochs': epochs,
            'config': config
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/ml/train",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def generate_with_ml(self, model_path: str, n_samples: int, temperature: float = 0.8) -> Dict:
        """Generate ODEs using ML model"""
        payload = {
            'model_path': model_path,
            'n_samples': n_samples,
            'temperature': temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/ml/generate",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None

# Initialize API client
api_client = ODEAPIClient(API_BASE_URL, API_KEY)

def render_ode_latex(ode_latex: str, solution_latex: str = None):
    """Render ODE and solution using LaTeX"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**ODE Equation:**")
        st.latex(ode_latex)
    
    if solution_latex:
        with col2:
            st.markdown("**Solution:**")
            st.latex(solution_latex)

def plot_solution(solution_str: str, x_range: tuple = (-5, 5), params: Dict = None):
    """Plot ODE solution"""
    try:
        # Parse solution
        x = symbols('x')
        solution_expr = sp.sympify(solution_str)
        
        # Substitute parameters if provided
        if params:
            for param, value in params.items():
                solution_expr = solution_expr.subs(param, value)
        
        # Create numerical function
        solution_func = sp.lambdify(x, solution_expr, 'numpy')
        
        # Generate plot data
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = solution_func(x_vals)
        
        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Solution',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='ODE Solution Plot',
            xaxis_title='x',
            yaxis_title='y(x)',
            template='plotly_white',
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting solution: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 ODE Master Generator System</h1>
        <p>Advanced Ordinary Differential Equation Generation, Verification & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Dashboard", "⚡ Generate ODEs", "✅ Verify Solutions", 
         "📊 Analysis Suite", "🤖 ML Training", "🧪 ML Generation", 
         "📈 Statistics", "⚙️ Settings"]
    )
    
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "⚡ Generate ODEs":
        render_generation_page()
    elif page == "✅ Verify Solutions":
        render_verification_page()
    elif page == "📊 Analysis Suite":
        render_analysis_page()
    elif page == "🤖 ML Training":
        render_ml_training_page()
    elif page == "🧪 ML Generation":
        render_ml_generation_page()
    elif page == "📈 Statistics":
        render_statistics_page()
    elif page == "⚙️ Settings":
        render_settings_page()

def render_dashboard():
    """Render main dashboard"""
    st.title("System Dashboard")
    
    # Get statistics
    stats = api_client.get_statistics()
    
    if stats:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>System Status</h3>
                <h1>🟢 {}</h1>
            </div>
            """.format(stats.get('status', 'Unknown').upper()), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Generated</h3>
                <h1>{:,}</h1>
            </div>
            """.format(stats.get('total_generated', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Active Jobs</h3>
                <h1>{}</h1>
            </div>
            """.format(stats.get('active_jobs', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Available Generators</h3>
                <h1>{}</h1>
            </div>
            """.format(stats.get('available_generators', 0)), unsafe_allow_html=True)
        
        # Job statistics
        if 'job_statistics' in stats:
            st.subheader("Job Statistics")
            job_stats = stats['job_statistics']
            
            # Create pie chart
            fig = px.pie(
                values=[job_stats.get('completed', 0), job_stats.get('failed', 0), 
                       job_stats.get('pending', 0), job_stats.get('running', 0)],
                names=['Completed', 'Failed', 'Pending', 'Running'],
                title='Job Status Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent ODEs
    if st.session_state.generated_odes:
        st.subheader("Recently Generated ODEs")
        
        for i, ode in enumerate(st.session_state.generated_odes[-5:]):
            with st.expander(f"ODE {ode.get('id', i)}", expanded=False):
                render_ode_latex(
                    ode.get('ode_latex', ode.get('ode', '')),
                    ode.get('solution_latex', ode.get('solution', ''))
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Generator", ode.get('generator', 'Unknown'))
                with col2:
                    st.metric("Function", ode.get('function', 'Unknown'))
                with col3:
                    verified_status = "✅ Verified" if ode.get('verified', False) else "❌ Not Verified"
                    st.metric("Status", verified_status)

def render_generation_page():
    """Render ODE generation page"""
    st.title("⚡ Generate ODEs")
    
    # Get available generators and functions
    generators_data = api_client.get_generators()
    functions_data = api_client.get_functions()
    
    generators = generators_data.get('all', [])
    functions = functions_data.get('functions', [])
    
    if not generators or not functions:
        st.error("Unable to load generators or functions. Please check API connection.")
        return
    
    # Generation form
    with st.form("generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_generator = st.selectbox(
                "Select Generator",
                generators,
                help="Choose the ODE generator type"
            )
            
            selected_function = st.selectbox(
                "Select Function",
                functions,
                help="Choose the mathematical function"
            )
            
            count = st.number_input(
                "Number of ODEs",
                min_value=1,
                max_value=100,
                value=1,
                help="How many ODEs to generate"
            )
        
        with col2:
            st.markdown("### Parameters")
            
            # Parameter inputs
            params = {}
            params['alpha'] = st.slider("α (alpha)", -2.0, 2.0, 1.0, 0.1)
            params['beta'] = st.slider("β (beta)", 0.1, 2.0, 1.0, 0.1)
            params['M'] = st.slider("M", -1.0, 1.0, 0.0, 0.1)
            
            # Nonlinear parameters
            if 'N' in selected_generator:
                params['q'] = st.slider("q (power)", 2, 5, 2)
                params['v'] = st.slider("v (power)", 2, 5, 3)
            
            # Pantograph parameter
            if 'L4' in selected_generator or 'N6' in selected_generator:
                params['a'] = st.slider("a (pantograph)", 2, 5, 2)
            
            verify = st.checkbox("Verify solutions", value=True)
        
        submit = st.form_submit_button("Generate ODEs", type="primary")
    
    if submit:
        with st.spinner("Generating ODEs..."):
            # Call API
            result = api_client.generate_odes(
                selected_generator,
                selected_function,
                params,
                count,
                verify
            )
            
            if result and 'job_id' in result:
                st.session_state.current_job_id = result['job_id']
                st.info(f"Job created: {result['job_id']}")
                
                # Poll for results
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while True:
                    job_status = api_client.get_job_status(result['job_id'])
                    
                    if job_status:
                        progress = job_status.get('progress', 0)
                        progress_bar.progress(int(progress))
                        status_text.text(f"Status: {job_status.get('status', 'Unknown')}")
                        
                        if job_status.get('status') == 'completed':
                            # Display results
                            results = job_status.get('results', [])
                            st.success(f"Generated {len(results)} ODEs successfully!")
                            
                            # Store in session state
                            st.session_state.generated_odes.extend(results)
                            
                            # Display ODEs
                            for ode in results:
                                with st.expander(f"ODE {ode.get('id', '')}", expanded=True):
                                    # Display equation and solution
                                    render_ode_latex(
                                        ode.get('ode_latex', ode.get('ode', '')),
                                        ode.get('solution_latex', ode.get('solution', ''))
                                    )
                                    
                                    # Display properties
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Complexity", ode.get('complexity', 'N/A'))
                                    with col2:
                                        verified = "✅" if ode.get('verified', False) else "❌"
                                        st.metric("Verified", verified)
                                    with col3:
                                        st.metric("Generation Time", f"{ode.get('generation_time', 0):.3f}s")
                                    
                                    # Plot solution
                                    if ode.get('solution'):
                                        fig = plot_solution(
                                            ode['solution'],
                                            params=ode.get('parameters', {})
                                        )
                                        if fig:
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Initial conditions
                                    if 'initial_conditions' in ode:
                                        st.markdown("**Initial Conditions:**")
                                        for ic_key, ic_value in ode['initial_conditions'].items():
                                            st.code(f"{ic_key} = {ic_value}")
                            
                            break
                        
                        elif job_status.get('status') == 'failed':
                            st.error(f"Job failed: {job_status.get('error', 'Unknown error')}")
                            break
                    
                    time.sleep(1)

def render_verification_page():
    """Render ODE verification page"""
    st.title("✅ Verify ODE Solutions")
    
    tab1, tab2 = st.tabs(["Manual Verification", "Batch Verification"])
    
    with tab1:
        st.markdown("### Enter ODE and Solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ode_input = st.text_area(
                "ODE Equation",
                placeholder="e.g., y''(x) + y(x) = sin(x)",
                height=100
            )
        
        with col2:
            solution_input = st.text_area(
                "Proposed Solution",
                placeholder="e.g., C1*cos(x) + C2*sin(x) + sin(x)/2",
                height=100
            )
        
        if st.button("Verify Solution", type="primary"):
            if ode_input and solution_input:
                with st.spinner("Verifying..."):
                    result = api_client.verify_ode(ode_input, solution_input)
                    
                    if result:
                        if result.get('verified'):
                            st.markdown("""
                            <div class="success-box">
                                <h3>✅ Solution Verified!</h3>
                                <p>Confidence: {:.2%}</p>
                                <p>Method: {}</p>
                            </div>
                            """.format(
                                result.get('confidence', 0),
                                result.get('method', 'Unknown')
                            ), unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="error-box">
                                <h3>❌ Solution Not Verified</h3>
                                <p>The provided solution does not satisfy the ODE.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show details
                        with st.expander("Verification Details"):
                            st.json(result.get('details', {}))
    
    with tab2:
        st.markdown("### Verify Generated ODEs")
        
        if st.session_state.generated_odes:
            # Create dataframe
            df_data = []
            for ode in st.session_state.generated_odes:
                df_data.append({
                    'ID': ode.get('id', ''),
                    'Generator': ode.get('generator', ''),
                    'Function': ode.get('function', ''),
                    'Verified': '✅' if ode.get('verified', False) else '❌',
                    'Complexity': ode.get('complexity', 0)
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Re-verify button
            if st.button("Re-verify All"):
                progress_bar = st.progress(0)
                
                for i, ode in enumerate(st.session_state.generated_odes):
                    if ode.get('ode') and ode.get('solution'):
                        result = api_client.verify_ode(
                            ode['ode'],
                            ode['solution']
                        )
                        
                        if result:
                            ode['verified'] = result.get('verified', False)
                            ode['verification_confidence'] = result.get('confidence', 0)
                    
                    progress_bar.progress((i + 1) / len(st.session_state.generated_odes))
                
                st.success("Re-verification complete!")
                st.rerun()
        else:
            st.info("No ODEs generated yet. Go to the Generation page to create some ODEs.")

def render_analysis_page():
    """Render analysis page"""
    st.title("📊 ODE Analysis Suite")
    
    if not st.session_state.generated_odes:
        st.warning("No ODEs available for analysis. Please generate some ODEs first.")
        return
    
    # Prepare data for analysis
    ode_list = [ode.get('ode', '') for ode in st.session_state.generated_odes if ode.get('ode')]
    
    if st.button("Run Comprehensive Analysis", type="primary"):
        with st.spinner("Analyzing ODEs..."):
            result = api_client.analyze_dataset(ode_list)
            
            if result and 'job_id' in result:
                # Poll for results
                while True:
                    job_status = api_client.get_job_status(result['job_id'])
                    
                    if job_status and job_status.get('status') == 'completed':
                        analysis_results = job_status.get('results', {})
                        st.session_state.analysis_results = analysis_results
                        break
                    
                    time.sleep(1)
    
    # Display analysis results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total ODEs", results.get('total_odes', 0))
        
        with col2:
            stats = results.get('statistics', {})
            st.metric("Verification Rate", f"{stats.get('verified_rate', 0):.1%}")
        
        with col3:
            st.metric("Avg Complexity", f"{stats.get('avg_complexity', 0):.1f}")
        
        with col4:
            st.metric("Complexity Std", f"{stats.get('complexity_std', 0):.1f}")
        
        # Generator distribution
        if 'generator_distribution' in results:
            st.subheader("Generator Distribution")
            
            gen_dist = results['generator_distribution']
            fig = px.bar(
                x=list(gen_dist.keys()),
                y=list(gen_dist.values()),
                title="ODEs by Generator",
                labels={'x': 'Generator', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Complexity distribution
        st.subheader("Complexity Analysis")
        
        complexities = [ode.get('complexity', 0) for ode in st.session_state.generated_odes]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=complexities,
            nbinsx=30,
            name='Complexity Distribution'
        ))
        fig.update_layout(
            title='ODE Complexity Distribution',
            xaxis_title='Complexity Score',
            yaxis_title='Count'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Verification analysis
        st.subheader("Verification Analysis")
        
        verified_count = sum(1 for ode in st.session_state.generated_odes if ode.get('verified', False))
        unverified_count = len(st.session_state.generated_odes) - verified_count
        
        fig = px.pie(
            values=[verified_count, unverified_count],
            names=['Verified', 'Not Verified'],
            title='Verification Status',
            color_discrete_map={'Verified': '#28a745', 'Not Verified': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_ml_training_page():
    """Render ML training page"""
    st.title("🤖 Machine Learning Training")
    
    st.markdown("""
    Train machine learning models on your ODE dataset to learn patterns and generate novel equations.
    """)
    
    # Model configuration
    with st.form("ml_training_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["pattern_net", "transformer", "vae"],
                help="Select the ML architecture"
            )
            
            epochs = st.number_input(
                "Training Epochs",
                min_value=1,
                max_value=1000,
                value=50
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                [16, 32, 64, 128],
                index=1
            )
        
        with col2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.00001, 0.0001, 0.001, 0.01, 0.1],
                value=0.001,
                format_func=lambda x: f"{x:.5f}"
            )
            
            early_stopping = st.checkbox("Early Stopping", value=True)
            
            # Advanced options
            with st.expander("Advanced Configuration"):
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
                hidden_dims = st.text_input("Hidden Dimensions", "256,128,64")
        
        # Dataset selection
        st.markdown("### Training Dataset")
        
        dataset_option = st.radio(
            "Dataset Source",
            ["Use Generated ODEs", "Upload Dataset"]
        )
        
        if dataset_option == "Use Generated ODEs":
            if st.session_state.generated_odes:
                st.info(f"Using {len(st.session_state.generated_odes)} generated ODEs")
                dataset_id = "session_odes"
            else:
                st.warning("No ODEs generated. Please generate ODEs first.")
                dataset_id = None
        else:
            uploaded_file = st.file_uploader("Upload JSONL dataset", type=['jsonl'])
            dataset_id = "uploaded_dataset" if uploaded_file else None
        
        submit = st.form_submit_button("Start Training", type="primary")
    
    if submit and dataset_id:
        # Prepare configuration
        config = {
            'dropout_rate': dropout_rate,
            'hidden_dims': [int(x.strip()) for x in hidden_dims.split(',')],
            'early_stopping': early_stopping
        }
        
        with st.spinner("Initializing training..."):
            result = api_client.train_model(
                dataset_id,
                model_type,
                epochs,
                config
            )
            
            if result and 'job_id' in result:
                st.info(f"Training job started: {result['job_id']}")
                
                # Training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()
                
                while True:
                    job_status = api_client.get_job_status(result['job_id'])
                    
                    if job_status:
                        progress = job_status.get('progress', 0)
                        progress_bar.progress(int(progress))
                        
                        # Update status
                        metadata = job_status.get('metadata', {})
                        status_text.text(
                            f"Epoch {metadata.get('current_epoch', 0)}/{metadata.get('total_epochs', epochs)}"
                        )
                        
                        if job_status.get('status') == 'completed':
                            results = job_status.get('results', {})
                            
                            st.success("Training completed successfully!")
                            
                            # Save model info
                            st.session_state.trained_models.append({
                                'model_id': results.get('model_id'),
                                'model_path': results.get('model_path'),
                                'model_type': model_type,
                                'metrics': results.get('final_metrics', {}),
                                'timestamp': datetime.now()
                            })
                            
                            # Display final metrics
                            st.subheader("Training Results")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Final Loss", f"{results.get('final_metrics', {}).get('loss', 0):.4f}")
                            with col2:
                                st.metric("Accuracy", f"{results.get('final_metrics', {}).get('accuracy', 0):.2%}")
                            with col3:
                                st.metric("Val Loss", f"{results.get('final_metrics', {}).get('validation_loss', 0):.4f}")
                            with col4:
                                st.metric("Val Accuracy", f"{results.get('final_metrics', {}).get('validation_accuracy', 0):.2%}")
                            
                            break
                        
                        elif job_status.get('status') == 'failed':
                            st.error(f"Training failed: {job_status.get('error', 'Unknown error')}")
                            break
                    
                    time.sleep(2)

def render_ml_generation_page():
    """Render ML generation page"""
    st.title("🧪 ML-Based ODE Generation")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train a model first.")
        return
    
    st.markdown("Generate novel ODEs using trained machine learning models.")
    
    # Model selection
    model_options = [
        f"{m['model_type']} - {m['model_id']}" 
        for m in st.session_state.trained_models
    ]
    
    selected_model_idx = st.selectbox(
        "Select Trained Model",
        range(len(model_options)),
        format_func=lambda x: model_options[x]
    )
    
    selected_model = st.session_state.trained_models[selected_model_idx]
    
    # Display model info
    with st.expander("Model Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", selected_model['model_type'])
            st.metric("Training Accuracy", f"{selected_model['metrics'].get('accuracy', 0):.2%}")
        with col2:
            st.metric("Validation Accuracy", f"{selected_model['metrics'].get('validation_accuracy', 0):.2%}")
            st.metric("Training Date", selected_model['timestamp'].strftime("%Y-%m-%d %H:%M"))
    
    # Generation parameters
    with st.form("ml_generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.number_input(
                "Number of ODEs to Generate",
                min_value=1,
                max_value=100,
                value=10
            )
            
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Higher values = more creative/diverse outputs"
            )
        
        with col2:
            target_generator = st.selectbox(
                "Target Generator Style (Optional)",
                ["None"] + api_client.get_generators().get('all', [])
            )
            
            target_function = st.selectbox(
                "Target Function Type (Optional)",
                ["None"] + api_client.get_functions().get('functions', [])
            )
            
            complexity_min = st.number_input("Min Complexity", value=50)
            complexity_max = st.number_input("Max Complexity", value=200)
        
        submit = st.form_submit_button("Generate ODEs", type="primary")
    
    if submit:
        with st.spinner("Generating novel ODEs..."):
            # Prepare request
            generation_params = {
                'model_path': selected_model['model_path'],
                'n_samples': n_samples,
                'temperature': temperature
            }
            
            if target_generator != "None":
                generation_params['generator'] = target_generator
            if target_function != "None":
                generation_params['function'] = target_function
            
            generation_params['complexity_range'] = [complexity_min, complexity_max]
            
            # Call API
            result = api_client.generate_with_ml(
                selected_model['model_path'],
                n_samples,
                temperature
            )
            
            if result and 'job_id' in result:
                # Poll for results
                progress_bar = st.progress(0)
                
                while True:
                    job_status = api_client.get_job_status(result['job_id'])
                    
                    if job_status:
                        progress = job_status.get('progress', 0)
                        progress_bar.progress(int(progress))
                        
                        if job_status.get('status') == 'completed':
                            ml_results = job_status.get('results', {})
                            
                            st.success(f"Generated {len(ml_results.get('odes', []))} novel ODEs!")
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Novelty Score", f"{ml_results.get('avg_novelty_score', 0):.2f}")
                            with col2:
                                st.metric("Diversity Score", f"{ml_results.get('diversity_score', 0):.2f}")
                            with col3:
                                st.metric("Valid Rate", f"{ml_results.get('valid_rate', 0):.1%}")
                            
                            # Display generated ODEs
                            st.subheader("Generated ODEs")
                            
                            for i, ode in enumerate(ml_results.get('odes', [])):
                                with st.expander(f"ML-Generated ODE {i+1}", expanded=i==0):
                                    if 'ode_latex' in ode:
                                        st.latex(ode['ode_latex'])
                                    else:
                                        st.code(ode.get('ode', ''))
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Novelty Score", f"{ode.get('novelty_score', 0):.2f}")
                                    with col2:
                                        valid = "✅" if ode.get('valid', False) else "❌"
                                        st.metric("Valid", valid)
                                    with col3:
                                        st.metric("Temperature", f"{ode.get('temperature', temperature):.1f}")
                            
                            break
                        
                        elif job_status.get('status') == 'failed':
                            st.error(f"Generation failed: {job_status.get('error', 'Unknown error')}")
                            break
                    
                    time.sleep(1)

def render_statistics_page():
    """Render statistics page"""
    st.title("📈 System Statistics")
    
    # Get statistics
    stats = api_client.get_statistics()
    
    if not stats:
        st.error("Unable to load statistics")
        return
    
    # System overview
    st.subheader("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        uptime_hours = stats.get('uptime', 0) / 3600
        st.metric("Uptime", f"{uptime_hours:.1f} hours")
    
    with col2:
        st.metric("Redis Status", "🟢 Connected" if stats.get('redis_available') else "🔴 Disconnected")
    
    with col3:
        st.metric("Generators", "🟢 Loaded" if stats.get('generators_available') else "🔴 Not Available")
    
    with col4:
        st.metric("Cache Size", stats.get('cache_size', 0))
    
    # Job statistics
    if 'job_statistics' in stats:
        st.subheader("Job Statistics")
        
        job_stats = stats['job_statistics']
        
        # Create metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Jobs", job_stats.get('total', 0))
        with col2:
            st.metric("Completed", job_stats.get('completed', 0))
        with col3:
            st.metric("Failed", job_stats.get('failed', 0))
        with col4:
            st.metric("Pending", job_stats.get('pending', 0))
        with col5:
            st.metric("Running", job_stats.get('running', 0))
        
        # Success rate
        if job_stats.get('total', 0) > 0:
            success_rate = job_stats.get('completed', 0) / job_stats.get('total', 0)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = success_rate * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Job Success Rate (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Generator performance
    st.subheader("Generator Performance")
    
    if st.session_state.generated_odes:
        # Aggregate by generator
        gen_performance = {}
        
        for ode in st.session_state.generated_odes:
            gen = ode.get('generator', 'Unknown')
            if gen not in gen_performance:
                gen_performance[gen] = {
                    'count': 0,
                    'verified': 0,
                    'total_time': 0,
                    'complexities': []
                }
            
            gen_performance[gen]['count'] += 1
            if ode.get('verified', False):
                gen_performance[gen]['verified'] += 1
            gen_performance[gen]['total_time'] += ode.get('generation_time', 0)
            gen_performance[gen]['complexities'].append(ode.get('complexity', 0))
        
        # Create performance table
        perf_data = []
        for gen, data in gen_performance.items():
            perf_data.append({
                'Generator': gen,
                'Count': data['count'],
                'Verified': data['verified'],
                'Success Rate': f"{100 * data['verified'] / data['count']:.1f}%",
                'Avg Time': f"{data['total_time'] / data['count']:.3f}s",
                'Avg Complexity': f"{np.mean(data['complexities']):.1f}"
            })
        
        df = pd.DataFrame(perf_data)
        st.dataframe(df, use_container_width=True)

def render_settings_page():
    """Render settings page"""
    st.title("⚙️ Settings")
    
    st.markdown("### API Configuration")
    
    with st.form("api_settings"):
        api_url = st.text_input(
            "API Base URL",
            value=API_BASE_URL,
            help="The base URL of your deployed API"
        )
        
        api_key = st.text_input(
            "API Key",
            value=API_KEY,
            type="password",
            help="Your API authentication key"
        )
        
        if st.form_submit_button("Update API Settings"):
            # In a real app, you'd save these securely
            st.success("API settings updated!")
    
    st.markdown("### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Generated ODEs"):
            if st.session_state.generated_odes:
                # Create JSON export
                export_data = {
                    'export_date': datetime.now().isoformat(),
                    'total_odes': len(st.session_state.generated_odes),
                    'odes': st.session_state.generated_odes
                }
                
                json_str = json.dumps(export_data, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="ode_export.json">Download JSON</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No ODEs to export")
    
    with col2:
        if st.button("Export Analysis Report"):
            if st.session_state.analysis_results:
                # Create report
                report = {
                    'report_date': datetime.now().isoformat(),
                    'analysis': st.session_state.analysis_results
                }
                
                json_str = json.dumps(report, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="analysis_report.json">Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No analysis results to export")
    
    st.markdown("### Clear Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Generated ODEs", type="secondary"):
            st.session_state.generated_odes = []
            st.success("Generated ODEs cleared")
    
    with col2:
        if st.button("Clear Trained Models", type="secondary"):
            st.session_state.trained_models = []
            st.success("Trained models cleared")
    
    with col3:
        if st.button("Clear All Data", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("All data cleared")
            st.rerun()

if __name__ == "__main__":
    main()
