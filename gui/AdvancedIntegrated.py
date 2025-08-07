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
import os
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ODE Master Generator - Advanced Interface",
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
    .author-credit {
        text-align: center;
        color: #666;
        font-style: italic;
        margin-top: -1rem;
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
        border-left: 4px solid #3498db;
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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        color: #856404;
    }
    .latex-equation {
        font-size: 1.2em;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        overflow-x: auto;
        margin: 0.5rem 0;
        font-family: 'Computer Modern', serif;
    }
    .doc-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .workflow-step {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2ecc71;
    }
    .progress-indicator {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .dataset-info {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'https://ode-api-production.up.railway.app')
API_KEY = os.getenv('API_KEY', 'test-key')

# Initialize session state
if 'generated_odes' not in st.session_state:
    st.session_state.generated_odes = []
if 'batch_dataset' not in st.session_state:
    st.session_state.batch_dataset = []
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'ml_generated_odes' not in st.session_state:
    st.session_state.ml_generated_odes = []
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = {
        'dataset_created': False,
        'model_trained': False,
        'current_model': None
    }

class ODEAPIClient:
    """Enhanced API client for ODE generation system"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
        self.timeout = 30
    
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
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API Error in generate_odes: {e}")
            st.error(f"API Error: {str(e)}")
            return None
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get job status with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}/api/v1/jobs/{job_id}",
                    headers=self.headers,
                    timeout=10
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return None
                else:
                    time.sleep(1)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get job status: {e}")
                    return None
                time.sleep(1)
        return None
    
    def create_dataset(self, odes: List[Dict], dataset_name: Optional[str] = None) -> Dict:
        """Create a dataset from ODEs"""
        payload = {
            'odes': odes,
            'dataset_name': dataset_name
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/datasets/create",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            st.error(f"Failed to create dataset: {str(e)}")
            return None
    
    def list_datasets(self) -> Dict:
        """List available datasets"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/datasets",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return {'datasets': [], 'count': 0}
    
    def train_ml_model(self, dataset: str, model_type: str, epochs: int, config: Dict) -> Dict:
        """Train ML model on dataset"""
        payload = {
            'dataset': dataset,
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': config.get('batch_size', 32),
            'learning_rate': config.get('learning_rate', 0.001),
            'early_stopping': config.get('early_stopping', True),
            'config': config
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/ml/train",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error training model: {e}")
            st.error(f"Failed to train model: {str(e)}")
            return None
    
    def generate_with_ml(self, model_path: str, n_samples: int, temperature: float = 0.8, **kwargs) -> Dict:
        """Generate ODEs using trained ML model"""
        payload = {
            'model_path': model_path,
            'n_samples': n_samples,
            'temperature': temperature
        }
        payload.update(kwargs)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/ml/generate",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in ML generation: {e}")
            st.error(f"Failed to generate with ML: {str(e)}")
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
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Verification Error: {e}")
            return None
    
    def get_generators(self) -> Dict:
        """Get available generators"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/generators",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting generators: {e}")
            return {
                'linear': ['L1', 'L2', 'L3', 'L4'],
                'nonlinear': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7'],
                'all': ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
            }
    
    def get_functions(self) -> Dict:
        """Get available functions"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/functions",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting functions: {e}")
            return {
                'functions': [
                    'identity', 'quadratic', 'cubic', 'quartic', 'quintic',
                    'exponential', 'exp_scaled', 'exp_quadratic', 'exp_negative',
                    'sine', 'cosine', 'tangent_safe', 'sine_scaled', 'cosine_scaled',
                    'sinh', 'cosh', 'tanh', 'log_safe', 'log_shifted',
                    'rational_simple', 'rational_stable', 'exp_sin', 'gaussian'
                ]
            }
    
    def get_models(self) -> Dict:
        """Get available ML models"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/models",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return {'models': [], 'count': 0}
    
    def get_statistics(self) -> Dict:
        """Get API statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/stats",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'status': 'unknown', 'error': str(e)}

# Initialize API client
api_client = ODEAPIClient(API_BASE_URL, API_KEY)

# Helper functions
def render_ode_latex(ode_str: str, solution_str: str = None, title: str = "ODE"):
    """Render ODE and solution using LaTeX formatting"""
    try:
        # Clean up the ODE string for display
        if 'Eq(' in ode_str:
            parts = ode_str.replace('Eq(', '').replace(')', '', -1).split(',', 1)
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                
                # Format derivatives
                lhs = lhs.replace('Derivative(y(x), (x, 2))', "y''(x)")
                lhs = lhs.replace('Derivative(y(x), x, 2)', "y''(x)")
                lhs = lhs.replace('Derivative(y(x), x)', "y'(x)")
                
                formatted_ode = f"{lhs} = {rhs}"
            else:
                formatted_ode = ode_str
        else:
            formatted_ode = ode_str
        
        # Display ODE
        st.markdown(f"**{title}:**")
        st.markdown(f'<div class="latex-equation">{formatted_ode}</div>', unsafe_allow_html=True)
        
        # Display solution if provided
        if solution_str:
            st.markdown("**Solution:**")
            formatted_solution = f"y(x) = {solution_str}"
            st.markdown(f'<div class="latex-equation">{formatted_solution}</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.code(ode_str, language='python')
        if solution_str:
            st.code(f"y(x) = {solution_str}", language='python')

def plot_solution(solution_str: str, x_range: tuple = (-5, 5), params: Dict = None):
    """Plot ODE solution"""
    try:
        import sympy as sp
        
        x = sp.Symbol('x')
        solution_expr = sp.sympify(solution_str)
        
        # Substitute parameters
        if params:
            for param, value in params.items():
                if param in ['alpha', 'beta', 'M']:
                    solution_expr = solution_expr.subs(param, value)
        
        # Create numerical function
        solution_func = sp.lambdify(x, solution_expr, 'numpy')
        
        # Generate plot data
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = solution_func(x_vals)
        
        # Handle infinities
        y_vals = np.where(np.abs(y_vals) > 1e10, np.nan, y_vals)
        
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
        logger.error(f"Error plotting solution: {e}")
        return None

def wait_for_job_completion(job_id: str, max_attempts: int = 60, poll_interval: int = 2) -> Optional[Dict]:
    """Wait for job completion with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for attempt in range(max_attempts):
        job_status = api_client.get_job_status(job_id)
        
        if job_status:
            progress = job_status.get('progress', 0)
            status = job_status.get('status', 'Unknown')
            
            progress_bar.progress(int(progress))
            
            # Update status with metadata if available
            metadata = job_status.get('metadata', {})
            if metadata:
                if 'current_epoch' in metadata:
                    status_text.text(
                        f"Status: {status} - Epoch {metadata['current_epoch']}/{metadata.get('total_epochs', '?')}"
                    )
                elif 'current' in metadata:
                    status_text.text(
                        f"Status: {status} - Processing {metadata['current']}/{metadata.get('total', '?')}"
                    )
                else:
                    status_text.text(f"Status: {status}")
            else:
                status_text.text(f"Status: {status}")
            
            if status == 'completed':
                progress_bar.progress(100)
                return job_status
            elif status == 'failed':
                st.error(f"Job failed: {job_status.get('error', 'Unknown error')}")
                return None
        
        time.sleep(poll_interval)
    
    st.error("Job timed out")
    return None

def create_parameter_controls(generator_type: str, key_prefix: str = "") -> Dict[str, float]:
    """Create parameter input controls based on generator type"""
    params = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        params['alpha'] = st.slider(
            "α (alpha)", -2.0, 2.0, 1.0, 0.1, 
            key=f"{key_prefix}_alpha"
        )
        params['beta'] = st.slider(
            "β (beta)", 0.1, 2.0, 1.0, 0.1,
            key=f"{key_prefix}_beta"
        )
    
    with col2:
        params['M'] = st.slider(
            "M", -1.0, 1.0, 0.0, 0.1,
            key=f"{key_prefix}_M"
        )
        
        # Add nonlinear parameters if needed
        if 'N' in generator_type:
            params['q'] = st.slider(
                "q (power)", 2, 5, 2,
                key=f"{key_prefix}_q"
            )
            params['v'] = st.slider(
                "v (power)", 2, 5, 3,
                key=f"{key_prefix}_v"
            )
        
        # Add pantograph parameter if needed
        if generator_type in ['L4', 'N6']:
            params['a'] = st.slider(
                "a (pantograph)", 2, 5, 2,
                key=f"{key_prefix}_a"
            )
    
    return params

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 ODE Master Generator System</h1>
        <p>Advanced Ordinary Differential Equation Generation, Verification & ML Analysis Platform</p>
    </div>
    <div class="author-credit">
        Created by Mohammad Abu Ghuwaleh
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Dashboard", "⚡ Quick Generate", "🔄 Complete Workflow", 
         "📦 Batch Generation", "🤖 ML Training", "🧪 ML Generation",
         "📊 Analysis", "📚 Documentation"]
    )
    
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "⚡ Quick Generate":
        render_quick_generate()
    elif page == "🔄 Complete Workflow":
        render_complete_workflow()
    elif page == "📦 Batch Generation":
        render_batch_generation()
    elif page == "🤖 ML Training":
        render_ml_training()
    elif page == "🧪 ML Generation":
        render_ml_generation()
    elif page == "📊 Analysis":
        render_analysis()
    elif page == "📚 Documentation":
        render_documentation()

def render_dashboard():
    """Render main dashboard with system overview"""
    st.title("System Dashboard")
    
    # Get statistics
    stats = api_client.get_statistics()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>System Status</h3>
            <h1>🟢 ONLINE</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_odes = len(st.session_state.generated_odes) + len(st.session_state.batch_dataset)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Generated</h3>
            <h1>{total_odes:,}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        dataset_created = "✅" if st.session_state.workflow_state['dataset_created'] else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Dataset Ready</h3>
            <h1>{dataset_created}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        model_trained = "✅" if st.session_state.workflow_state['model_trained'] else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Model Trained</h3>
            <h1>{model_trained}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Workflow status
    st.subheader("Workflow Status")
    
    workflow_steps = [
        ("1. Generate Batch ODEs", len(st.session_state.batch_dataset) > 0),
        ("2. Create Dataset", st.session_state.workflow_state['dataset_created']),
        ("3. Train ML Model", st.session_state.workflow_state['model_trained']),
        ("4. Generate with ML", len(st.session_state.ml_generated_odes) > 0)
    ]
    
    for step, completed in workflow_steps:
        status = "✅" if completed else "⏳"
        st.markdown(f"{status} **{step}**")
    
    # Recent activity
    if st.session_state.batch_dataset:
        st.subheader("Recent Batch Generation")
        recent_odes = st.session_state.batch_dataset[-5:]
        
        for i, ode in enumerate(recent_odes):
            with st.expander(f"ODE {ode.get('id', i)}", expanded=False):
                render_ode_latex(
                    ode.get('ode', ''),
                    ode.get('solution', '')
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Generator", ode.get('generator', 'Unknown'))
                with col2:
                    st.metric("Function", ode.get('function', 'Unknown'))
                with col3:
                    verified = "✅" if ode.get('verified', False) else "❌"
                    st.metric("Verified", verified)

def render_quick_generate():
    """Quick single ODE generation"""
    st.title("⚡ Quick ODE Generation")
    
    # Get available options
    generators_data = api_client.get_generators()
    functions_data = api_client.get_functions()
    
    col1, col2 = st.columns(2)
    
    with col1:
        generator = st.selectbox(
            "Select Generator",
            generators_data.get('all', []),
            help="Choose the ODE generator type"
        )
    
    with col2:
        function = st.selectbox(
            "Select Function",
            functions_data.get('functions', []),
            help="Choose the mathematical function"
        )
    
    # Parameters
    st.subheader("Parameters")
    params = create_parameter_controls(generator, "quick")
    
    verify = st.checkbox("Verify solution", value=True)
    
    if st.button("Generate ODE", type="primary"):
        with st.spinner("Generating ODE..."):
            result = api_client.generate_odes(
                generator, function, params, count=1, verify=verify
            )
            
            if result and 'job_id' in result:
                job_status = wait_for_job_completion(result['job_id'])
                
                if job_status and job_status.get('results'):
                    st.success("ODE generated successfully!")
                    
                    ode = job_status['results'][0]
                    st.session_state.generated_odes.append(ode)
                    
                    # Display ODE
                    st.markdown("---")
                    render_ode_latex(
                        ode.get('ode', ''),
                        ode.get('solution', ''),
                        "Generated ODE"
                    )
                    
                    # Display properties
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Complexity", ode.get('complexity', 'N/A'))
                    with col2:
                        verified = "✅" if ode.get('verified', False) else "❌"
                        st.metric("Verified", verified)
                    with col3:
                        confidence = ode.get('properties', {}).get('verification_confidence', 0)
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Plot solution
                    if ode.get('solution'):
                        st.subheader("Solution Plot")
                        fig = plot_solution(ode['solution'], params=params)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

def render_complete_workflow():
    """Complete workflow from generation to ML"""
    st.title("🔄 Complete ODE Workflow")
    st.markdown("Follow this guided workflow to generate ODEs, train ML models, and generate novel equations.")
    
    # Step 1: Batch Generation
    with st.expander("Step 1: Generate Batch Dataset", expanded=not st.session_state.batch_dataset):
        st.markdown("""
        <div class="workflow-step">
            <h3>📦 Generate Training Dataset</h3>
            <p>Create a diverse dataset of ODEs for ML training</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.batch_dataset:
            generators_data = api_client.get_generators()
            functions_data = api_client.get_functions()
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_generators = st.multiselect(
                    "Select Generators",
                    generators_data.get('all', []),
                    default=['L1', 'L2', 'N1', 'N2']
                )
            
            with col2:
                selected_functions = st.multiselect(
                    "Select Functions",
                    functions_data.get('functions', [])[:10],
                    default=['sine', 'exponential', 'quadratic', 'gaussian']
                )
            
            samples_per_combo = st.slider("Samples per combination", 1, 10, 3)
            
            total_combinations = len(selected_generators) * len(selected_functions) * samples_per_combo
            st.info(f"This will generate {total_combinations} ODEs")
            
            if st.button("Generate Batch Dataset", type="primary"):
                batch_generate_workflow(selected_generators, selected_functions, samples_per_combo)
        else:
            st.success(f"✅ Batch dataset ready with {len(st.session_state.batch_dataset)} ODEs")
            
            # Show summary
            df = pd.DataFrame(st.session_state.batch_dataset)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total ODEs", len(df))
            with col2:
                verified_count = df['verified'].sum() if 'verified' in df else 0
                st.metric("Verified", verified_count)
            with col3:
                unique_combos = df[['generator', 'function']].drop_duplicates().shape[0]
                st.metric("Unique Combinations", unique_combos)
            
            if st.button("Clear Dataset"):
                st.session_state.batch_dataset = []
                st.session_state.workflow_state['dataset_created'] = False
                st.rerun()
    
    # Step 2: Create Dataset
    with st.expander("Step 2: Create Training Dataset", 
                     expanded=(len(st.session_state.batch_dataset) > 0 and 
                              not st.session_state.workflow_state['dataset_created'])):
        st.markdown("""
        <div class="workflow-step">
            <h3>💾 Save Dataset for Training</h3>
            <p>Create a named dataset from your generated ODEs</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.batch_dataset and not st.session_state.workflow_state['dataset_created']:
            dataset_name = st.text_input(
                "Dataset Name",
                value=f"ode_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if st.button("Create Dataset", type="primary"):
                with st.spinner("Creating dataset..."):
                    result = api_client.create_dataset(
                        st.session_state.batch_dataset,
                        dataset_name
                    )
                    
                    if result:
                        st.success(f"Dataset '{dataset_name}' created successfully!")
                        st.session_state.current_dataset = dataset_name
                        st.session_state.workflow_state['dataset_created'] = True
                        st.rerun()
        
        elif st.session_state.workflow_state['dataset_created']:
            st.success(f"✅ Dataset '{st.session_state.current_dataset}' is ready for training")
        else:
            st.warning("⚠️ Generate a batch dataset first (Step 1)")
    
    # Step 3: Train ML Model
    with st.expander("Step 3: Train ML Model", 
                     expanded=(st.session_state.workflow_state['dataset_created'] and 
                              not st.session_state.workflow_state['model_trained'])):
        st.markdown("""
        <div class="workflow-step">
            <h3>🤖 Train Machine Learning Model</h3>
            <p>Train a neural network to learn ODE patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.workflow_state['dataset_created']:
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Model Type",
                    ["pattern_net", "transformer", "vae"],
                    format_func=lambda x: {
                        "pattern_net": "PatternNet - Fast Training",
                        "transformer": "Transformer - Advanced",
                        "vae": "VAE - Generative"
                    }[x]
                )
                
                epochs = st.slider("Training Epochs", 10, 100, 50)
            
            with col2:
                batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.001, 0.01],
                    value=0.001,
                    format_func=lambda x: f"{x:.4f}"
                )
            
            # Model-specific parameters
            config = {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'early_stopping': st.checkbox("Early Stopping", value=True)
            }
            
            if model_type == "pattern_net":
                config['hidden_dims'] = [256, 128, 64]
                config['dropout_rate'] = 0.2
            elif model_type == "vae":
                config['latent_dim'] = st.slider("Latent Dimension", 32, 128, 64)
                config['beta'] = st.slider("KL Weight (β)", 0.1, 2.0, 1.0)
            
            if st.button("Start Training", type="primary"):
                train_model_workflow(model_type, epochs, config)
        
        elif st.session_state.workflow_state['model_trained']:
            st.success(f"✅ Model trained: {st.session_state.workflow_state['current_model']}")
            
            # Show training results
            if st.session_state.trained_models:
                latest_model = st.session_state.trained_models[-1]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Type", latest_model['model_type'])
                with col2:
                    st.metric("Accuracy", f"{latest_model['accuracy']:.2%}")
                with col3:
                    st.metric("Training Time", f"{latest_model['training_time']:.1f}s")
        else:
            st.warning("⚠️ Create a dataset first (Step 2)")
    
    # Step 4: Generate with ML
    with st.expander("Step 4: Generate Novel ODEs with ML",
                     expanded=st.session_state.workflow_state['model_trained']):
        st.markdown("""
        <div class="workflow-step">
            <h3>🧪 Generate Novel ODEs</h3>
            <p>Use your trained model to generate new equations</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.workflow_state['model_trained']:
            col1, col2 = st.columns(2)
            
            with col1:
                n_samples = st.slider("Number of ODEs to Generate", 5, 50, 10)
                temperature = st.slider("Temperature (Creativity)", 0.5, 1.5, 0.8, 0.1)
            
            with col2:
                target_generator = st.selectbox(
                    "Target Generator Style",
                    ["Auto"] + api_client.get_generators().get('all', [])
                )
                target_function = st.selectbox(
                    "Target Function Type",
                    ["Auto"] + api_client.get_functions().get('functions', [])[:10]
                )
            
            if st.button("Generate Novel ODEs", type="primary"):
                generate_with_ml_workflow(n_samples, temperature, target_generator, target_function)
        else:
            st.warning("⚠️ Train a model first (Step 3)")
    
    # Display ML-generated ODEs
    if st.session_state.ml_generated_odes:
        st.markdown("---")
        st.subheader("🎨 ML-Generated ODEs")
        
        for i, ode in enumerate(st.session_state.ml_generated_odes[-5:]):
            with st.expander(f"Novel ODE {ode.get('id', i)}", expanded=i==0):
                render_ode_latex(
                    ode.get('ode', ''),
                    ode.get('solution', ''),
                    "ML-Generated ODE"
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model", ode.get('model_type', 'Unknown'))
                with col2:
                    st.metric("Temperature", ode.get('temperature', 'N/A'))
                with col3:
                    verified = "✅" if ode.get('verified', False) else "❌"
                    st.metric("Valid", verified)
                with col4:
                    st.metric("Complexity", ode.get('complexity', 'N/A'))

def batch_generate_workflow(generators: List[str], functions: List[str], samples: int):
    """Execute batch generation workflow"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_results = []
    total_tasks = len(generators) * len(functions) * samples
    completed = 0
    
    # Parameter ranges for diversity
    param_ranges = {
        'alpha': [-2.0, -1.0, 0.0, 1.0, 2.0],
        'beta': [0.5, 1.0, 1.5, 2.0],
        'M': [-0.5, 0.0, 0.5]
    }
    
    for generator in generators:
        for function in functions:
            for sample_idx in range(samples):
                completed += 1
                progress_bar.progress(completed / total_tasks)
                status_text.text(f"Generating: {generator} + {function} ({completed}/{total_tasks})")
                
                # Sample parameters
                params = {
                    'alpha': np.random.choice(param_ranges['alpha']),
                    'beta': np.random.choice(param_ranges['beta']),
                    'M': np.random.choice(param_ranges['M']),
                    'q': np.random.choice([2, 3, 4]),
                    'v': np.random.choice([2, 3, 4]),
                    'a': 2
                }
                
                # Generate ODE
                result = api_client.generate_odes(
                    generator, function, params, count=1, verify=True
                )
                
                if result and 'job_id' in result:
                    job_status = wait_for_job_completion(result['job_id'], max_attempts=30, poll_interval=1)
                    
                    if job_status and job_status.get('results'):
                        ode = job_status['results'][0]
                        ode['batch_id'] = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        ode['sample_idx'] = sample_idx
                        batch_results.append(ode)
    
    # Store results
    st.session_state.batch_dataset.extend(batch_results)
    
    # Show summary
    st.success(f"Batch generation complete! Generated {len(batch_results)} ODEs")
    
    verified_count = sum(1 for ode in batch_results if ode.get('verified', False))
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Generated", len(batch_results))
    with col2:
        st.metric("Verified", verified_count)
    with col3:
        st.metric("Success Rate", f"{100 * verified_count / len(batch_results):.1f}%" if batch_results else "0%")

def train_model_workflow(model_type: str, epochs: int, config: Dict):
    """Execute model training workflow"""
    with st.spinner("Starting training job..."):
        result = api_client.train_ml_model(
            st.session_state.current_dataset,
            model_type,
            epochs,
            config
        )
        
        if result and 'job_id' in result:
            st.info(f"Training job started: {result['job_id']}")
            
            # Monitor training progress
            job_status = wait_for_job_completion(result['job_id'], max_attempts=300, poll_interval=3)
            
            if job_status and job_status.get('status') == 'completed':
                results = job_status.get('results', {})
                
                st.success("Model training completed!")
                
                # Save model info
                model_info = {
                    'model_id': results.get('model_id'),
                    'model_path': results.get('model_path'),
                    'model_type': model_type,
                    'accuracy': results.get('final_metrics', {}).get('accuracy', 0),
                    'training_time': results.get('training_time', 0),
                    'timestamp': datetime.now()
                }
                
                st.session_state.trained_models.append(model_info)
                st.session_state.workflow_state['model_trained'] = True
                st.session_state.workflow_state['current_model'] = model_info['model_id']
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Loss", f"{results.get('final_metrics', {}).get('loss', 0):.4f}")
                with col2:
                    st.metric("Accuracy", f"{results.get('final_metrics', {}).get('accuracy', 0):.2%}")
                with col3:
                    st.metric("Val Loss", f"{results.get('final_metrics', {}).get('validation_loss', 0):.4f}")
                with col4:
                    st.metric("Training Time", f"{results.get('training_time', 0):.1f}s")
                
                st.rerun()

def generate_with_ml_workflow(n_samples: int, temperature: float, target_generator: str, target_function: str):
    """Execute ML generation workflow"""
    if not st.session_state.trained_models:
        st.error("No trained models available")
        return
    
    latest_model = st.session_state.trained_models[-1]
    
    with st.spinner("Generating novel ODEs..."):
        kwargs = {}
        if target_generator != "Auto":
            kwargs['generator'] = target_generator
        if target_function != "Auto":
            kwargs['function'] = target_function
        
        result = api_client.generate_with_ml(
            latest_model['model_path'],
            n_samples,
            temperature,
            **kwargs
        )
        
        if result and 'job_id' in result:
            job_status = wait_for_job_completion(result['job_id'], max_attempts=60)
            
            if job_status and job_status.get('status') == 'completed':
                results = job_status.get('results', {})
                generated_odes = results.get('odes', [])
                
                st.success(f"Generated {len(generated_odes)} novel ODEs!")
                
                # Add metadata
                for ode in generated_odes:
                    ode['model_type'] = latest_model['model_type']
                    ode['model_id'] = latest_model['model_id']
                    ode['timestamp'] = datetime.now().isoformat()
                
                # Store results
                st.session_state.ml_generated_odes.extend(generated_odes)
                
                # Show metrics
                metrics = results.get('metrics', {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Valid ODEs", metrics.get('valid_count', 0))
                with col2:
                    st.metric("Verified", metrics.get('verified_count', 0))
                with col3:
                    diversity = metrics.get('diversity_metrics', {})
                    st.metric("Diversity", f"{diversity.get('uniqueness_ratio', 0):.2f}")
                
                st.rerun()

def render_batch_generation():
    """Dedicated batch generation page"""
    st.title("📦 Batch ODE Generation")
    
    generators_data = api_client.get_generators()
    functions_data = api_client.get_functions()
    
    # Configuration
    st.subheader("Batch Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_generators = st.multiselect(
            "Generators",
            generators_data.get('all', []),
            default=generators_data.get('all', [])[:4]
        )
    
    with col2:
        selected_functions = st.multiselect(
            "Functions",
            functions_data.get('functions', []),
            default=functions_data.get('functions', [])[:6]
        )
    
    # Parameter configuration
    st.subheader("Parameter Ranges")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alpha_values = st.multiselect(
            "α values",
            [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
            default=[0.0, 1.0]
        )
    
    with col2:
        beta_values = st.multiselect(
            "β values",
            [0.5, 1.0, 1.5, 2.0],
            default=[1.0]
        )
    
    with col3:
        m_values = st.multiselect(
            "M values",
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            default=[0.0]
        )
    
    samples_per_combo = st.slider("Samples per combination", 1, 5, 1)
    
    total_combinations = (
        len(selected_generators) * len(selected_functions) * 
        len(alpha_values) * len(beta_values) * len(m_values) * samples_per_combo
    )
    
    st.info(f"This will generate approximately {total_combinations} ODEs")
    
    if st.button("Generate Batch", type="primary"):
        batch_generate_detailed(
            selected_generators, selected_functions,
            alpha_values, beta_values, m_values,
            samples_per_combo
        )

def batch_generate_detailed(generators, functions, alphas, betas, ms, samples):
    """Detailed batch generation with all parameter combinations"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_results = []
    total_tasks = len(generators) * len(functions) * len(alphas) * len(betas) * len(ms) * samples
    completed = 0
    
    for generator in generators:
        for function in functions:
            for alpha in alphas:
                for beta in betas:
                    for m in ms:
                        for sample_idx in range(samples):
                            completed += 1
                            progress_bar.progress(completed / total_tasks)
                            status_text.text(
                                f"Generating: {generator} + {function} "
                                f"(α={alpha}, β={beta}, M={m}) "
                                f"({completed}/{total_tasks})"
                            )
                            
                            params = {
                                'alpha': float(alpha),
                                'beta': float(beta),
                                'M': float(m),
                                'q': 2,
                                'v': 3,
                                'a': 2
                            }
                            
                            result = api_client.generate_odes(
                                generator, function, params, count=1, verify=True
                            )
                            
                            if result and 'job_id' in result:
                                job_status = wait_for_job_completion(
                                    result['job_id'], 
                                    max_attempts=20, 
                                    poll_interval=0.5
                                )
                                
                                if job_status and job_status.get('results'):
                                    ode = job_status['results'][0]
                                    ode['batch_params'] = params
                                    batch_results.append(ode)
    
    st.session_state.batch_dataset.extend(batch_results)
    
    st.success(f"Generated {len(batch_results)} ODEs!")
    
    # Display results summary
    if batch_results:
        df = pd.DataFrame(batch_results)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Generated", len(df))
        with col2:
            verified = df['verified'].sum() if 'verified' in df else 0
            st.metric("Verified", verified)
        with col3:
            st.metric("Success Rate", f"{100 * verified / len(df):.1f}%")
        
        # Show distribution
        st.subheader("Generation Distribution")
        
        fig = px.sunburst(
            df,
            path=['generator', 'function'],
            title="ODEs by Generator and Function"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_ml_training():
    """ML Training page"""
    st.title("🤖 Machine Learning Training")
    
    # Check for datasets
    datasets = api_client.list_datasets()
    
    if not datasets['datasets']:
        st.warning("No datasets available. Please create a dataset first.")
        if st.button("Go to Batch Generation"):
            st.session_state.page = "📦 Batch Generation"
            st.rerun()
        return
    
    # Dataset selection
    dataset_names = [d['name'] for d in datasets['datasets']]
    selected_dataset = st.selectbox("Select Dataset", dataset_names)
    
    # Find dataset info
    dataset_info = next((d for d in datasets['datasets'] if d['name'] == selected_dataset), None)
    if dataset_info:
        st.markdown(f"""
        <div class="dataset-info">
            <b>Dataset:</b> {dataset_info['name']}<br>
            <b>Size:</b> {dataset_info.get('size', 'Unknown')} ODEs<br>
            <b>Created:</b> {dataset_info.get('created_at', 'Unknown')}
        </div>
        """, unsafe_allow_html=True)
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["pattern_net", "transformer", "vae"],
            format_func=lambda x: {
                "pattern_net": "PatternNet - Pattern Recognition",
                "transformer": "Transformer - Sequence Model",
                "vae": "VAE - Generative Model"
            }[x]
        )
        
        epochs = st.number_input("Training Epochs", 10, 200, 50)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    with col2:
        learning_rate = st.select_slider(
            "Learning Rate",
            [0.00001, 0.0001, 0.001, 0.01],
            value=0.001,
            format_func=lambda x: f"{x:.5f}"
        )
        
        early_stopping = st.checkbox("Early Stopping", value=True)
        
        # Model-specific parameters
        config = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'early_stopping': early_stopping
        }
        
        if model_type == "pattern_net":
            config['dropout_rate'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
        elif model_type == "vae":
            config['latent_dim'] = st.slider("Latent Dimension", 16, 256, 64)
            config['beta'] = st.slider("KL Weight (β)", 0.1, 10.0, 1.0)
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Initializing training..."):
            result = api_client.train_ml_model(
                selected_dataset,
                model_type,
                epochs,
                config
            )
            
            if result and 'job_id' in result:
                st.info(f"Training job started: {result['job_id']}")
                
                # Monitor training
                job_status = wait_for_job_completion(
                    result['job_id'], 
                    max_attempts=epochs * 5,
                    poll_interval=2
                )
                
                if job_status and job_status.get('status') == 'completed':
                    results = job_status.get('results', {})
                    
                    st.success("Training completed!")
                    
                    # Save model info
                    model_info = {
                        'model_id': results.get('model_id'),
                        'model_path': results.get('model_path'),
                        'model_type': model_type,
                        'dataset': selected_dataset,
                        'config': config,
                        'metrics': results.get('final_metrics', {}),
                        'training_time': results.get('training_time', 0),
                        'timestamp': datetime.now()
                    }
                    
                    st.session_state.trained_models.append(model_info)
                    
                    # Display results
                    st.subheader("Training Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Loss", f"{results.get('final_metrics', {}).get('loss', 0):.4f}")
                    with col2:
                        st.metric("Accuracy", f"{results.get('final_metrics', {}).get('accuracy', 0):.2%}")
                    with col3:
                        st.metric("Val Loss", f"{results.get('final_metrics', {}).get('validation_loss', 0):.4f}")
                    with col4:
                        st.metric("Time", f"{results.get('training_time', 0):.1f}s")

def render_ml_generation():
    """ML Generation page"""
    st.title("🧪 ML-Based ODE Generation")
    
    # Check for trained models
    models_response = api_client.get_models()
    available_models = models_response.get('models', [])
    
    if not available_models and not st.session_state.trained_models:
        st.warning("No trained models available. Please train a model first.")
        if st.button("Go to ML Training"):
            st.session_state.page = "🤖 ML Training"
            st.rerun()
        return
    
    # Model selection
    st.subheader("Select Model")
    
    model_options = []
    for model in available_models:
        model_desc = f"{model['name']} - Created: {model.get('created', 'Unknown')}"
        model_options.append((model['path'], model_desc))
    
    if model_options:
        selected_model_path = st.selectbox(
            "Available Models",
            [m[0] for m in model_options],
            format_func=lambda x: next(m[1] for m in model_options if m[0] == x)
        )
    else:
        st.error("No models found")
        return
    
    # Generation parameters
    st.subheader("Generation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Number of ODEs", 5, 100, 20)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    
    with col2:
        generators_data = api_client.get_generators()
        functions_data = api_client.get_functions()
        
        target_generator = st.selectbox(
            "Target Generator Style",
            ["Auto"] + generators_data.get('all', [])
        )
        
        target_function = st.selectbox(
            "Target Function Type",
            ["Auto"] + functions_data.get('functions', [])[:10]
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        complexity_min = st.number_input("Min Complexity", 10, 500, 50)
        complexity_max = st.number_input("Max Complexity", complexity_min, 1000, 200)
        verify_generated = st.checkbox("Verify Generated ODEs", value=True)
    
    if st.button("Generate Novel ODEs", type="primary"):
        with st.spinner("Generating novel ODEs..."):
            kwargs = {
                'complexity_range': [complexity_min, complexity_max]
            }
            
            if target_generator != "Auto":
                kwargs['generator'] = target_generator
            if target_function != "Auto":
                kwargs['function'] = target_function
            
            result = api_client.generate_with_ml(
                selected_model_path,
                n_samples,
                temperature,
                **kwargs
            )
            
            if result and 'job_id' in result:
                job_status = wait_for_job_completion(result['job_id'])
                
                if job_status and job_status.get('status') == 'completed':
                    results = job_status.get('results', {})
                    generated_odes = results.get('odes', [])
                    
                    st.success(f"Generated {len(generated_odes)} novel ODEs!")
                    
                    # Store results
                    st.session_state.ml_generated_odes.extend(generated_odes)
                    
                    # Display results
                    st.subheader("Generated ODEs")
                    
                    for i, ode in enumerate(generated_odes[:10]):
                        with st.expander(f"Novel ODE {i+1}", expanded=(i < 3)):
                            render_ode_latex(
                                ode.get('ode', ''),
                                ode.get('solution', ''),
                                "Generated ODE"
                            )
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Temperature", ode.get('temperature', 'N/A'))
                            with col2:
                                verified = "✅" if ode.get('verified', False) else "❌"
                                st.metric("Valid", verified)
                            with col3:
                                st.metric("Complexity", ode.get('complexity', 'N/A'))

def render_analysis():
    """Analysis page"""
    st.title("📊 ODE Dataset Analysis")
    
    all_odes = st.session_state.generated_odes + st.session_state.batch_dataset + st.session_state.ml_generated_odes
    
    if not all_odes:
        st.warning("No ODEs available for analysis. Generate some ODEs first.")
        return
    
    st.info(f"Analyzing {len(all_odes)} ODEs")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_odes)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total ODEs", len(df))
    
    with col2:
        verified = df['verified'].sum() if 'verified' in df else 0
        st.metric("Verified", verified)
    
    with col3:
        ver_rate = verified / len(df) if len(df) > 0 else 0
        st.metric("Verification Rate", f"{ver_rate:.1%}")
    
    with col4:
        if 'complexity' in df:
            avg_complexity = df['complexity'].mean()
            st.metric("Avg Complexity", f"{avg_complexity:.1f}")
        else:
            st.metric("Avg Complexity", "N/A")
    
    # Visualizations
    st.subheader("Generator Distribution")
    if 'generator' in df:
        fig = px.bar(
            df['generator'].value_counts(),
            title="ODEs by Generator",
            labels={'index': 'Generator', 'value': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Function Distribution")
    if 'function' in df:
        top_functions = df['function'].value_counts().head(10)
        fig = px.bar(
            top_functions,
            title="Top 10 Functions",
            labels={'index': 'Function', 'value': 'Count'}
