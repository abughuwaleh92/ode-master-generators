# gui/integrated_interface.py
"""
Complete Integrated Interface for ODE Master Generator System
Covers all API endpoints and provides full workflow from generation to ML
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import base64
from io import BytesIO, StringIO
import os
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import aiohttp
import zipfile
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ODE Master Generator - Integrated Interface",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-card h3 {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-card h1 {
        color: #1e293b;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .ode-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    .success-box {
        background-color: #d1fae5;
        border: 1px solid #6ee7b7;
        padding: 1rem;
        border-radius: 8px;
        color: #065f46;
    }
    .error-box {
        background-color: #fee2e2;
        border: 1px solid #fca5a5;
        padding: 1rem;
        border-radius: 8px;
        color: #991b1b;
    }
    .info-box {
        background-color: #dbeafe;
        border: 1px solid #93c5fd;
        padding: 1rem;
        border-radius: 8px;
        color: #1e3a8a;
    }
    .warning-box {
        background-color: #fef3c7;
        border: 1px solid #fcd34d;
        padding: 1rem;
        border-radius: 8px;
        color: #92400e;
    }
    .section-header {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    .latex-equation {
        font-size: 1.1em;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        overflow-x: auto;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
        font-family: 'Computer Modern', 'Times New Roman', serif;
    }
    .dataset-info {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
    }
    .workflow-step {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .progress-indicator {
        background: #eff6ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #bfdbfe;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .status-online {
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
    }
    .status-offline {
        background-color: #ef4444;
        box-shadow: 0 0 10px #ef4444;
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
if 'available_datasets' not in st.session_state:
    st.session_state.available_datasets = []
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'ml_generated_odes' not in st.session_state:
    st.session_state.ml_generated_odes = []
if 'api_status' not in st.session_state:
    st.session_state.api_status = None
if 'available_generators' not in st.session_state:
    st.session_state.available_generators = []
if 'available_functions' not in st.session_state:
    st.session_state.available_functions = []

class ODEAPIClient:
    """Complete API client for ODE generation system"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
        self.timeout = 30
    
    async def check_health(self) -> Dict:
        """Check API health status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return {"status": "error", "message": f"Status code: {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
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
    
    def batch_generate(self, generators: List[str], functions: List[str], 
                      samples_per_combination: int, parameters: Optional[Dict] = None,
                      verify: bool = True, dataset_name: Optional[str] = None) -> Dict:
        """Batch generate ODEs"""
        payload = {
            'generators': generators,
            'functions': functions,
            'samples_per_combination': samples_per_combination,
            'verify': verify
        }
        
        if parameters:
            payload['parameters'] = parameters
        if dataset_name:
            payload['dataset_name'] = dataset_name
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/batch_generate",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API Error in batch_generate: {e}")
            st.error(f"API Error: {str(e)}")
            return None
    
    def verify_ode(self, ode: str, solution: str, method: str = "substitution") -> Dict:
        """Verify ODE solution"""
        payload = {
            'ode': ode,
            'solution': solution,
            'method': method
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
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get job status"""
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
                return {"status": "error", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {"status": "error", "error": str(e)}
    
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
                'linear': [],
                'nonlinear': [],
                'all': []
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
                'functions': [],
                'categories': {}
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
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        try:
            response = requests.get(
                f"{self.base_url}/metrics",
                headers={'X-API-Key': self.headers['X-API-Key']},
                timeout=10
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return ""

# Initialize API client
api_client = ODEAPIClient(API_BASE_URL, API_KEY)

# Helper functions
def render_ode_latex(ode_str: str, solution_str: str = None, title: str = "ODE"):
    """Render ODE and solution using LaTeX formatting"""
    try:
        # Clean up the ODE string for display
        formatted_ode = ode_str
        if 'Eq(' in ode_str:
            parts = ode_str.replace('Eq(', '').replace(')', '', -1).split(',', 1)
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                
                # Format derivatives
                lhs = lhs.replace('Derivative(y(x), (x, 2))', "y''(x)")
                lhs = lhs.replace('Derivative(y(x), x, 2)', "y''(x)")
                lhs = lhs.replace('Derivative(y(x), x)', "y'(x)")
                lhs = lhs.replace('Derivative(y(x), (x, 3))', "y'''(x)")
                
                formatted_ode = f"{lhs} = {rhs}"
        
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
        
        # Parse solution
        solution_expr = sp.sympify(solution_str)
        
        # Substitute parameters if provided
        if params:
            for param, value in params.items():
                if param in str(solution_expr):
                    solution_expr = solution_expr.subs(param, value)
        
        # Create numerical function
        solution_func = sp.lambdify(x, solution_expr, 'numpy')
        
        # Generate plot data
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = solution_func(x_vals)
        
        # Handle infinities and complex values
        y_vals = np.real(y_vals)  # Take real part if complex
        y_vals = np.where(np.abs(y_vals) > 1e10, np.nan, y_vals)
        
        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Solution',
            line=dict(color='#3b82f6', width=2)
        ))
        
        fig.update_layout(
            title='ODE Solution Plot',
            xaxis_title='x',
            yaxis_title='y(x)',
            template='plotly_white',
            height=400,
            hovermode='x unified'
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
        
        if job_status and 'status' in job_status:
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
                        f"Status: {status} - Processing {metadata['current']}/{metadata.get('total', '?')} - {metadata.get('status', '')}"
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
            key=f"{key_prefix}_alpha",
            help="Shift parameter in the function argument"
        )
        params['beta'] = st.slider(
            "β (beta)", 0.1, 2.0, 1.0, 0.1,
            key=f"{key_prefix}_beta",
            help="Scale parameter in the exponential term"
        )
    
    with col2:
        params['M'] = st.slider(
            "M", -1.0, 1.0, 0.0, 0.1,
            key=f"{key_prefix}_M",
            help="Constant term in the solution"
        )
        
        # Add nonlinear parameters if needed
        if 'N' in generator_type:
            params['q'] = st.slider(
                "q (power)", 2, 5, 2,
                key=f"{key_prefix}_q",
                help="Power of the highest derivative term"
            )
            if generator_type in ['N2', 'N3', 'N6', 'N7']:
                params['v'] = st.slider(
                    "v (power)", 2, 5, 3,
                    key=f"{key_prefix}_v",
                    help="Power of the first derivative term"
                )
        
        # Add pantograph parameter if needed
        if generator_type in ['L4', 'N6']:
            params['a'] = st.slider(
                "a (pantograph)", 2.0, 5.0, 2.0, 0.5,
                key=f"{key_prefix}_a",
                help="Delay factor for pantograph term y(x/a)"
            )
    
    return params

def download_dataset(dataset: List[Dict], filename: str = "ode_dataset.jsonl"):
    """Create download link for dataset"""
    # Convert to JSONL
    jsonl_content = "\n".join(json.dumps(ode) for ode in dataset)
    
    # Create download link
    b64 = base64.b64encode(jsonl_content.encode()).decode()
    href = f'<a href="data:application/jsonl;base64,{b64}" download="{filename}">📥 Download Dataset ({len(dataset)} ODEs)</a>'
    st.markdown(href, unsafe_allow_html=True)

def export_analysis_report(analysis_data: Dict, odes: List[Dict]):
    """Export comprehensive analysis report"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_odes": len(odes),
            "verified": sum(1 for ode in odes if ode.get('verified', False)),
            "generators_used": list(set(ode.get('generator', 'Unknown') for ode in odes)),
            "functions_used": list(set(ode.get('function', 'Unknown') for ode in odes))
        },
        "analysis": analysis_data,
        "sample_odes": odes[:5] if len(odes) > 5 else odes
    }
    
    # Create download link
    json_str = json.dumps(report, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="ode_analysis_report.json">📊 Download Analysis Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# Main application
async def check_api_status():
    """Check API status asynchronously"""
    status = await api_client.check_health()
    st.session_state.api_status = status
    
    if status.get('status') == 'healthy':
        # Get available generators and functions
        generators_data = api_client.get_generators()
        functions_data = api_client.get_functions()
        
        st.session_state.available_generators = generators_data.get('all', [])
        st.session_state.available_functions = functions_data.get('functions', [])
    
    return status

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 ODE Master Generator System</h1>
        <p>Complete Integrated Interface for ODE Generation, Verification & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API status
    if st.session_state.api_status is None:
        asyncio.run(check_api_status())
    
    # API Status indicator in sidebar
    with st.sidebar:
        st.markdown("### System Status")
        if st.session_state.api_status and st.session_state.api_status.get('status') == 'healthy':
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-online"></span>
                <span style="color: #10b981; font-weight: bold;">API Online</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Display system info
            st.caption(f"Generators: {len(st.session_state.available_generators)}")
            st.caption(f"Functions: {len(st.session_state.available_functions)}")
            st.caption(f"ML: {'✅' if st.session_state.api_status.get('ml_enabled') else '❌'}")
        else:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <span class="status-indicator status-offline"></span>
                <span style="color: #ef4444; font-weight: bold;">API Offline</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Retry Connection"):
                asyncio.run(check_api_status())
                st.rerun()
    
    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Dashboard", "⚡ Quick Generate", "📦 Batch Generation", 
         "✅ Verification", "📊 Dataset Management", "🤖 ML Training", 
         "🧪 ML Generation", "📈 Analysis & Visualization", "🛠️ System Tools",
         "📚 Documentation"]
    )
    
    # Load available datasets and models
    if not st.session_state.available_datasets:
        datasets_response = api_client.list_datasets()
        st.session_state.available_datasets = datasets_response.get('datasets', [])
    
    if not st.session_state.available_models:
        models_response = api_client.get_models()
        st.session_state.available_models = models_response.get('models', [])
    
    # Render selected page
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "⚡ Quick Generate":
        render_quick_generate()
    elif page == "📦 Batch Generation":
        render_batch_generation()
    elif page == "✅ Verification":
        render_verification()
    elif page == "📊 Dataset Management":
        render_dataset_management()
    elif page == "🤖 ML Training":
        render_ml_training()
    elif page == "🧪 ML Generation":
        render_ml_generation()
    elif page == "📈 Analysis & Visualization":
        render_analysis()
    elif page == "🛠️ System Tools":
        render_system_tools()
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
        status_color = "#10b981" if stats.get('status') == 'operational' else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <h3>System Status</h3>
            <h1 style="color: {status_color};">{'🟢 ONLINE' if stats.get('status') == 'operational' else '🔴 OFFLINE'}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_generated = stats.get('total_generated_24h', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Generated (24h)</h3>
            <h1>{total_generated:,}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        verification_rate = stats.get('verification_success_rate', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Verification Rate</h3>
            <h1>{verification_rate:.1%}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        active_jobs = stats.get('active_jobs', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Active Jobs</h3>
            <h1>{active_jobs}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Generator Performance
    st.markdown('<div class="section-header">Generator Performance</div>', unsafe_allow_html=True)
    
    generator_stats = stats.get('generator_performance', {})
    if generator_stats:
        # Create performance dataframe
        perf_data = []
        for gen_name, gen_stats in generator_stats.items():
            perf_data.append({
                'Generator': gen_name,
                'Success Rate': f"{gen_stats.get('success_rate', 0):.1%}",
                'Avg Time (s)': f"{gen_stats.get('avg_time', 0):.3f}",
                'Total Generated': gen_stats.get('total_generated', 0),
                'Verified': gen_stats.get('total_verified', 0)
            })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    # Recent Activity
    st.markdown('<div class="section-header">Recent Activity</div>', unsafe_allow_html=True)
    
    if st.session_state.generated_odes:
        recent_odes = st.session_state.generated_odes[-5:]
        
        for i, ode in enumerate(recent_odes):
            with st.expander(f"Recent ODE {i+1}: {ode.get('generator', 'Unknown')} + {ode.get('function', 'Unknown')}", expanded=False):
                render_ode_latex(
                    ode.get('ode', ''),
                    ode.get('solution', '')
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Complexity", ode.get('complexity', 'N/A'))
                with col2:
                    verified = "✅" if ode.get('verified', False) else "❌"
                    st.metric("Verified", verified)
                with col3:
                    st.metric("Time", f"{ode.get('properties', {}).get('generation_time_ms', 0):.0f}ms")
    else:
        st.info("No recent ODEs generated. Use Quick Generate or Batch Generation to create ODEs.")
    
    # System Information
    st.markdown('<div class="section-header">System Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h4>Available Resources</h4>
            <ul>
                <li>Generators: {len(st.session_state.available_generators)}</li>
                <li>Functions: {len(st.session_state.available_functions)}</li>
                <li>Datasets: {len(st.session_state.available_datasets)}</li>
                <li>ML Models: {len(st.session_state.available_models)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>System Capabilities</h4>
            <ul>
                <li>ML Features: {'✅ Enabled' if stats.get('ml_enabled') else '❌ Disabled'}</li>
                <li>Redis Cache: {'✅ Active' if stats.get('redis_enabled') else '❌ Inactive'}</li>
                <li>API Version: v2.0.0</li>
                <li>Max Batch Size: 1000 ODEs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_quick_generate():
    """Quick single ODE generation"""
    st.title("⚡ Quick ODE Generation")
    st.markdown("Generate individual ODEs with custom parameters and instant verification.")
    
    if not st.session_state.available_generators:
        st.error("No generators available. Please check API connection.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        generator = st.selectbox(
            "Select Generator",
            st.session_state.available_generators,
            help="Choose the ODE generator type",
            format_func=lambda x: f"{x} ({'Linear' if x.startswith('L') else 'Nonlinear'})"
        )
    
    with col2:
        # Group functions by category
        function_categories = {
            "Polynomial": ["identity", "quadratic", "cubic", "quartic", "quintic"],
            "Exponential": ["exponential", "exp_scaled", "exp_quadratic", "exp_negative"],
            "Trigonometric": ["sine", "cosine", "tangent_safe", "sine_scaled", "cosine_scaled"],
            "Hyperbolic": ["sinh", "cosh", "tanh"],
            "Other": []
        }
        
        # Categorize functions
        categorized = []
        for cat, funcs in function_categories.items():
            categorized.extend(funcs)
        
        # Add remaining functions to "Other"
        for func in st.session_state.available_functions:
            if func not in categorized:
                function_categories["Other"].append(func)
        
        # Create grouped selectbox
        function_options = []
        for category, funcs in function_categories.items():
            if funcs:
                function_options.extend([(f"{category}: {func}", func) for func in funcs])
        
        function = st.selectbox(
            "Select Function",
            options=[opt[1] for opt in function_options],
            format_func=lambda x: next((opt[0] for opt in function_options if opt[1] == x), x),
            help="Choose the mathematical function"
        )
    
    # Parameters
    st.markdown("### Parameters")
    params = create_parameter_controls(generator, "quick")
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            verify = st.checkbox("Verify solution", value=True)
            count = st.number_input("Number of ODEs", min_value=1, max_value=10, value=1)
        with col2:
            show_plot = st.checkbox("Plot solution", value=True)
            x_range = st.slider("Plot range", -10.0, 10.0, (-5.0, 5.0))
    
    if st.button("🚀 Generate ODE", type="primary"):
        with st.spinner("Generating ODE..."):
            result = api_client.generate_odes(
                generator, function, params, count=count, verify=verify
            )
            
            if result and 'job_id' in result:
                job_status = wait_for_job_completion(result['job_id'])
                
                if job_status and job_status.get('results'):
                    st.success(f"Successfully generated {len(job_status['results'])} ODE(s)!")
                    
                    for idx, ode in enumerate(job_status['results']):
                        st.session_state.generated_odes.append(ode)
                        
                        # Display ODE
                        st.markdown("---")
                        if count > 1:
                            st.subheader(f"ODE {idx + 1}")
                        
                        render_ode_latex(
                            ode.get('ode', ''),
                            ode.get('solution', ''),
                            "Generated ODE"
                        )
                        
                        # Display properties
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Complexity", ode.get('complexity', 'N/A'))
                        with col2:
                            verified = "✅ Verified" if ode.get('verified', False) else "❌ Not Verified"
                            st.metric("Status", verified)
                        with col3:
                            confidence = ode.get('properties', {}).get('verification_confidence', 0)
                            st.metric("Confidence", f"{confidence:.2%}")
                        with col4:
                            gen_time = ode.get('properties', {}).get('generation_time_ms', 0)
                            st.metric("Time", f"{gen_time:.0f}ms")
                        
                        # Additional properties
                        if st.checkbox(f"Show details {f'(ODE {idx + 1})' if count > 1 else ''}", key=f"details_{idx}"):
                            props = ode.get('properties', {})
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Structure Metrics:**")
                                st.write(f"- Operation Count: {props.get('operation_count', 'N/A')}")
                                st.write(f"- Atom Count: {props.get('atom_count', 'N/A')}")
                                st.write(f"- Symbol Count: {props.get('symbol_count', 'N/A')}")
                                st.write(f"- Has Pantograph: {'Yes' if props.get('has_pantograph', False) else 'No'}")
                            
                            with col2:
                                st.markdown("**Verification Details:**")
                                st.write(f"- Method: {props.get('verification_method', 'N/A')}")
                                st.write(f"- Initial Conditions: {props.get('initial_conditions', {})}")
                                st.write(f"- Parameters: {ode.get('parameters', {})}")
                        
                        # Plot solution
                        if show_plot and ode.get('solution'):
                            fig = plot_solution(ode['solution'], x_range, params)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not plot solution")
    
    # Export options
    if st.session_state.generated_odes:
        st.markdown("### Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download All Generated ODEs"):
                download_dataset(st.session_state.generated_odes, "quick_generated_odes.jsonl")
        
        with col2:
            if st.button("🗑️ Clear Generated ODEs"):
                st.session_state.generated_odes = []
                st.success("Cleared all generated ODEs")
                st.rerun()

def render_batch_generation():
    """Batch ODE generation page"""
    st.title("📦 Batch ODE Generation")
    st.markdown("Generate large datasets of ODEs for analysis and machine learning.")
    
    # Configuration tabs
    tab1, tab2, tab3 = st.tabs(["Basic Configuration", "Parameter Ranges", "Advanced Settings"])
    
    with tab1:
        st.markdown("### Select Generators and Functions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Group generators
            linear_gens = [g for g in st.session_state.available_generators if g.startswith('L')]
            nonlinear_gens = [g for g in st.session_state.available_generators if g.startswith('N')]
            
            st.markdown("**Linear Generators**")
            selected_linear = st.multiselect(
                "Select linear generators",
                linear_gens,
                default=linear_gens[:2] if len(linear_gens) >= 2 else linear_gens
            )
            
            st.markdown("**Nonlinear Generators**")
            selected_nonlinear = st.multiselect(
                "Select nonlinear generators",
                nonlinear_gens,
                default=nonlinear_gens[:2] if len(nonlinear_gens) >= 2 else nonlinear_gens
            )
            
            selected_generators = selected_linear + selected_nonlinear
        
        with col2:
            # Function selection with categories
            function_groups = {
                "Polynomial": ["identity", "quadratic", "cubic", "quartic", "quintic"],
                "Exponential": ["exponential", "exp_scaled", "exp_quadratic", "exp_negative"],
                "Trigonometric": ["sine", "cosine", "tangent_safe", "sine_scaled", "cosine_scaled"],
                "Hyperbolic": ["sinh", "cosh", "tanh"],
                "Logarithmic": ["log_safe", "log_shifted", "log_scaled"],
                "Rational": ["rational_simple", "rational_stable", "rational_cubic"],
                "Composite": ["exp_sin", "sin_exp", "gaussian", "bessel_like"]
            }
            
            selected_functions = []
            for group_name, group_funcs in function_groups.items():
                available_in_group = [f for f in group_funcs if f in st.session_state.available_functions]
                if available_in_group:
                    selected = st.multiselect(
                        f"{group_name} Functions",
                        available_in_group,
                        default=available_in_group[:2] if len(available_in_group) >= 2 else available_in_group,
                        key=f"batch_{group_name}"
                    )
                    selected_functions.extend(selected)
        
        samples_per_combo = st.slider(
            "Samples per combination",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of ODEs to generate for each generator-function pair"
        )
        
        total_combinations = len(selected_generators) * len(selected_functions) * samples_per_combo
        st.info(f"This will generate **{total_combinations:,}** ODEs ({len(selected_generators)} generators × {len(selected_functions)} functions × {samples_per_combo} samples)")
    
    with tab2:
        st.markdown("### Parameter Ranges")
        st.markdown("Define the parameter ranges for random sampling")
        
        param_ranges = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Parameters**")
            param_ranges['alpha'] = st.multiselect(
                "α (alpha) values",
                options=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
                default=[0.0, 1.0],
                help="Shift parameter values"
            )
            
            param_ranges['beta'] = st.multiselect(
                "β (beta) values",
                options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                default=[1.0, 2.0],
                help="Scale parameter values"
            )
            
            param_ranges['M'] = st.multiselect(
                "M values",
                options=[-1.0, -0.5, 0.0, 0.5, 1.0],
                default=[0.0],
                help="Constant term values"
            )
        
        with col2:
            st.markdown("**Nonlinear Parameters**")
            if any(g.startswith('N') for g in selected_generators):
                param_ranges['q'] = st.multiselect(
                    "q (power) values",
                    options=[2, 3, 4, 5],
                    default=[2, 3],
                    help="Power values for nonlinear terms"
                )
                
                param_ranges['v'] = st.multiselect(
                    "v (power) values",
                    options=[2, 3, 4, 5],
                    default=[2, 3],
                    help="Power values for first derivative"
                )
            
            if any(g in ['L4', 'N6'] for g in selected_generators):
                param_ranges['a'] = st.multiselect(
                    "a (pantograph) values",
                    options=[2.0, 2.5, 3.0, 3.5, 4.0],
                    default=[2.0],
                    help="Delay factor for pantograph terms"
                )
    
    with tab3:
        st.markdown("### Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            verify = st.checkbox("Verify all generated ODEs", value=True)
            save_dataset = st.checkbox("Save as dataset", value=True)
            
            if save_dataset:
                dataset_name = st.text_input(
                    "Dataset Name",
                    value=f"batch_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    help="Name for the saved dataset"
                )
        
        with col2:
            parallel_jobs = st.number_input(
                "Parallel jobs",
                min_value=1,
                max_value=10,
                value=4,
                help="Number of parallel generation jobs"
            )
            
            export_format = st.selectbox(
                "Export format",
                ["JSONL", "Parquet", "CSV"],
                help="Format for exported dataset"
            )
    
    # Generation button
    if st.button("🚀 Start Batch Generation", type="primary"):
        if not selected_generators or not selected_functions:
            st.error("Please select at least one generator and one function")
            return
        
        with st.spinner(f"Generating {total_combinations:,} ODEs..."):
            result = api_client.batch_generate(
                generators=selected_generators,
                functions=selected_functions,
                samples_per_combination=samples_per_combo,
                parameters=param_ranges if param_ranges else None,
                verify=verify,
                dataset_name=dataset_name if save_dataset else None
            )
            
            if result and 'job_id' in result:
                st.info(f"Batch generation job started. Expected: {result.get('total_expected', 0):,} ODEs")
                
                job_status = wait_for_job_completion(result['job_id'], max_attempts=300, poll_interval=3)
                
                if job_status and job_status.get('status') == 'completed':
                    results = job_status.get('results', {})
                    
                    st.success(f"Batch generation completed! Generated {results.get('total_generated', 0):,} ODEs")
                    
                    # Display summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Generated", f"{results.get('total_generated', 0):,}")
                    with col2:
                        st.metric("Verified", f"{results.get('verified_count', 0):,}")
                    with col3:
                        st.metric("Success Rate", f"{results.get('summary', {}).get('verified', 0) / results.get('total_generated', 1) * 100:.1f}%")
                    with col4:
                        avg_complexity = results.get('summary', {}).get('avg_complexity', 0)
                        st.metric("Avg Complexity", f"{avg_complexity:.1f}")
                    
                    # Update session state
                    if save_dataset and 'dataset_info' in results:
                        st.session_state.current_dataset = results['dataset_info']['name']
                        st.session_state.available_datasets.append(results['dataset_info'])
                        
                        st.markdown("### Dataset Created")
                        st.markdown(f"""
                        <div class="dataset-info">
                            <b>Name:</b> {results['dataset_info']['name']}<br>
                            <b>Size:</b> {results['dataset_info']['size']:,} ODEs<br>
                            <b>Path:</b> {results['dataset_info']['path']}<br>
                            <b>Generators:</b> {', '.join(results.get('generators_used', []))}<br>
                            <b>Functions:</b> {', '.join(results.get('functions_used', []))[:50]}...
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # If ODEs were returned directly (not saved as dataset)
                    if 'odes' in results:
                        st.session_state.batch_dataset = results['odes']
                        
                        # Show sample ODEs
                        st.markdown("### Sample Generated ODEs")
                        sample_odes = results['odes'][:5]
                        
                        for i, ode in enumerate(sample_odes):
                            with st.expander(f"Sample ODE {i+1}: {ode.get('generator_name', 'Unknown')} + {ode.get('function_name', 'Unknown')}", expanded=False):
                                render_ode_latex(
                                    ode.get('ode_symbolic', ''),
                                    ode.get('solution_symbolic', '')
                                )
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Complexity", ode.get('complexity_score', 'N/A'))
                                with col2:
                                    verified = "✅" if ode.get('verified', False) else "❌"
                                    st.metric("Verified", verified)
                                with col3:
                                    st.metric("Generation Time", f"{ode.get('generation_time', 0)*1000:.0f}ms")
                        
                        # Export options
                        st.markdown("### Export Dataset")
                        if st.button("📥 Download Generated Dataset"):
                            if export_format == "JSONL":
                                download_dataset(results['odes'], f"batch_odes_{len(results['odes'])}.jsonl")
                            elif export_format == "CSV":
                                df = pd.DataFrame(results['odes'])
                                csv = df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:text/csv;base64,{b64}" download="batch_odes.csv">📥 Download CSV</a>'
                                st.markdown(href, unsafe_allow_html=True)

def render_verification():
    """ODE verification page"""
    st.title("✅ ODE Verification")
    st.markdown("Verify ODE solutions using multiple methods")
    
    tab1, tab2 = st.tabs(["Single Verification", "Batch Verification"])
    
    with tab1:
        st.markdown("### Enter ODE and Solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ode_input = st.text_area(
                "ODE Equation",
                value="Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))",
                height=100,
                help="Enter the ODE in SymPy format"
            )
        
        with col2:
            solution_input = st.text_area(
                "Proposed Solution",
                value="pi*sin(x)",
                height=100,
                help="Enter the proposed solution"
            )
        
        verification_method = st.selectbox(
            "Verification Method",
            ["substitution", "numerical", "checkodesol"],
            help="Select the verification method"
        )
        
        if st.button("🔍 Verify Solution", type="primary"):
            with st.spinner("Verifying..."):
                result = api_client.verify_ode(ode_input, solution_input, verification_method)
                
                if result:
                    if result.get('verified'):
                        st.success(f"✅ Solution verified! Confidence: {result.get('confidence', 0):.2%}")
                    else:
                        st.error("❌ Solution could not be verified")
                    
                    # Display details
                    st.markdown("### Verification Details")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Method Used", result.get('method', 'Unknown'))
                        st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
                    
                    with col2:
                        details = result.get('details', {})
                        st.metric("Residual", details.get('residual', 'N/A'))
                        
                        if st.checkbox("Show raw result"):
                            st.json(result)
    
    with tab2:
        st.markdown("### Batch Verification")
        st.markdown("Upload a dataset to verify multiple ODEs")
        
        uploaded_file = st.file_uploader(
            "Choose a JSONL file",
            type=['jsonl', 'json'],
            help="Upload a JSONL file containing ODEs to verify"
        )
        
        if uploaded_file is not None:
            # Read file
            odes_to_verify = []
            for line in uploaded_file:
                if line.strip():
                    odes_to_verify.append(json.loads(line))
            
            st.info(f"Loaded {len(odes_to_verify)} ODEs from file")
            
            # Verification options
            col1, col2 = st.columns(2)
            
            with col1:
                verify_method = st.selectbox(
                    "Verification method",
                    ["substitution", "numerical", "all"],
                    key="batch_verify_method"
                )
            
            with col2:
                max_workers = st.number_input(
                    "Parallel workers",
                    min_value=1,
                    max_value=10,
                    value=4
                )
            
            if st.button("🔍 Start Batch Verification"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                verified_count = 0
                results = []
                
                for i, ode_data in enumerate(odes_to_verify):
                    progress_bar.progress((i + 1) / len(odes_to_verify))
                    status_text.text(f"Verifying ODE {i + 1}/{len(odes_to_verify)}")
                    
                    if 'ode_symbolic' in ode_data and 'solution_symbolic' in ode_data:
                        result = api_client.verify_ode(
                            ode_data['ode_symbolic'],
                            ode_data['solution_symbolic'],
                            verify_method
                        )
                        
                        if result and result.get('verified'):
                            verified_count += 1
                        
                        ode_data['verification_result'] = result
                        results.append(ode_data)
                
                st.success(f"Verification complete! {verified_count}/{len(odes_to_verify)} ODEs verified")
                
                # Show results summary
                st.markdown("### Verification Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total ODEs", len(odes_to_verify))
                with col2:
                    st.metric("Verified", verified_count)
                with col3:
                    st.metric("Success Rate", f"{verified_count/len(odes_to_verify)*100:.1f}%")
                
                # Download verified dataset
                if st.button("📥 Download Verified Dataset"):
                    download_dataset(results, "verified_odes.jsonl")

def render_dataset_management():
    """Dataset management page"""
    st.title("📊 Dataset Management")
    st.markdown("Manage your ODE datasets for training and analysis")
    
    # Refresh datasets
    if st.button("🔄 Refresh Datasets"):
        datasets_response = api_client.list_datasets()
        st.session_state.available_datasets = datasets_response.get('datasets', [])
        st.success("Datasets refreshed!")
    
    if not st.session_state.available_datasets:
        st.info("No datasets available. Generate some ODEs using Batch Generation first.")
        return
    
    # Dataset list
    st.markdown("### Available Datasets")
    
    for dataset in st.session_state.available_datasets:
        with st.expander(f"📁 {dataset['name']} ({dataset['size']:,} ODEs)", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                **Created:** {dataset.get('created_at', 'Unknown')}  
                **Path:** {dataset.get('path', 'Unknown')}  
                **Size:** {dataset.get('file_size_bytes', 0) / 1024 / 1024:.2f} MB
                """)
                
                if 'generators' in dataset:
                    st.markdown(f"**Generators:** {', '.join(dataset['generators'])}")
                if 'functions' in dataset:
                    st.markdown(f"**Functions:** {', '.join(dataset['functions'][:5])}...")
            
            with col2:
                if st.button(f"Select", key=f"select_{dataset['name']}"):
                    st.session_state.current_dataset = dataset['name']
                    st.success(f"Selected dataset: {dataset['name']}")
                
                if st.button(f"Download", key=f"download_{dataset['name']}"):
                    st.info(f"Download link for {dataset['name']} would be generated here")
    
    # Create new dataset
    st.markdown("### Create New Dataset")
    
    if st.session_state.batch_dataset:
        st.info(f"You have {len(st.session_state.batch_dataset)} ODEs ready to save as a dataset")
        
        new_dataset_name = st.text_input(
            "Dataset Name",
            value=f"custom_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
       
       if st.button("💾 Save as Dataset"):
           with st.spinner("Creating dataset..."):
               result = api_client.create_dataset(
                   st.session_state.batch_dataset,
                   new_dataset_name
               )
               
               if result:
                   st.success(f"Dataset '{new_dataset_name}' created successfully!")
                   st.session_state.current_dataset = new_dataset_name
                   
                   # Refresh datasets
                   datasets_response = api_client.list_datasets()
                   st.session_state.available_datasets = datasets_response.get('datasets', [])
                   
                   # Clear batch dataset
                   st.session_state.batch_dataset = []
   else:
       st.info("Generate ODEs using Batch Generation to create a new dataset")
   
   # Dataset operations
   if st.session_state.current_dataset:
       st.markdown(f"### Current Dataset: {st.session_state.current_dataset}")
       
       col1, col2, col3 = st.columns(3)
       
       with col1:
           if st.button("📈 Analyze Dataset"):
               st.info("Dataset analysis would be performed here")
       
       with col2:
           if st.button("🔧 Preprocess Dataset"):
               st.info("Dataset preprocessing options would be shown here")
       
       with col3:
           if st.button("🗑️ Delete Dataset"):
               if st.checkbox("Confirm deletion"):
                   st.warning(f"Dataset {st.session_state.current_dataset} would be deleted")

def render_ml_training():
   """ML Training page"""
   st.title("🤖 Machine Learning Training")
   st.markdown("Train ML models on your ODE datasets")
   
   if not st.session_state.available_datasets:
       st.warning("No datasets available. Please create a dataset first.")
       if st.button("Go to Dataset Management"):
           st.session_state.current_page = "📊 Dataset Management"
           st.rerun()
       return
   
   # Model configuration
   st.markdown("### Training Configuration")
   
   col1, col2 = st.columns(2)
   
   with col1:
       # Dataset selection
       dataset_names = [d['name'] for d in st.session_state.available_datasets]
       selected_dataset = st.selectbox(
           "Select Dataset",
           dataset_names,
           index=dataset_names.index(st.session_state.current_dataset) if st.session_state.current_dataset in dataset_names else 0
       )
       
       # Find dataset info
       dataset_info = next((d for d in st.session_state.available_datasets if d['name'] == selected_dataset), None)
       if dataset_info:
           st.markdown(f"""
           <div class="dataset-info">
               <b>Size:</b> {dataset_info.get('size', 'Unknown')} ODEs<br>
               <b>Created:</b> {dataset_info.get('created_at', 'Unknown')}
           </div>
           """, unsafe_allow_html=True)
       
       # Model type selection
       model_type = st.selectbox(
           "Model Type",
           ["pattern_net", "transformer", "vae"],
           format_func=lambda x: {
               "pattern_net": "PatternNet - Fast Pattern Recognition",
               "transformer": "Transformer - Advanced Sequence Model",
               "vae": "VAE - Generative Model"
           }[x],
           help="Choose the model architecture"
       )
   
   with col2:
       # Training parameters
       epochs = st.slider("Training Epochs", 10, 200, 50, 10)
       batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
       learning_rate = st.select_slider(
           "Learning Rate",
           options=[0.00001, 0.0001, 0.001, 0.01],
           value=0.001,
           format_func=lambda x: f"{x:.5f}"
       )
       early_stopping = st.checkbox("Early Stopping", value=True)
   
   # Model-specific parameters
   st.markdown("### Model-Specific Parameters")
   config = {
       'batch_size': batch_size,
       'learning_rate': learning_rate,
       'early_stopping': early_stopping
   }
   
   if model_type == "pattern_net":
       col1, col2 = st.columns(2)
       with col1:
           config['hidden_dims'] = st.multiselect(
               "Hidden Layer Dimensions",
               [64, 128, 256, 512],
               default=[256, 128, 64]
           )
       with col2:
           config['dropout_rate'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
   
   elif model_type == "transformer":
       col1, col2 = st.columns(2)
       with col1:
           config['d_model'] = st.selectbox("Model Dimension", [256, 512, 768], index=1)
           config['n_heads'] = st.selectbox("Attention Heads", [4, 8, 12], index=1)
       with col2:
           config['n_layers'] = st.slider("Number of Layers", 2, 12, 6)
           config['dim_feedforward'] = st.selectbox("Feedforward Dimension", [1024, 2048, 4096], index=1)
   
   elif model_type == "vae":
       col1, col2 = st.columns(2)
       with col1:
           config['latent_dim'] = st.slider("Latent Dimension", 16, 256, 64, 16)
           config['hidden_dim'] = st.slider("Hidden Dimension", 128, 512, 256, 32)
       with col2:
           config['beta'] = st.slider("KL Weight (β)", 0.1, 10.0, 1.0, 0.1)
   
   # Training visualization options
   with st.expander("Training Options"):
       show_live_metrics = st.checkbox("Show live training metrics", value=True)
       save_checkpoints = st.checkbox("Save model checkpoints", value=True)
       checkpoint_interval = st.number_input("Checkpoint interval (epochs)", 10, 50, 10) if save_checkpoints else None
   
   # Start training
   if st.button("🚀 Start Training", type="primary"):
       with st.spinner("Initializing training..."):
           result = api_client.train_ml_model(
               selected_dataset,
               model_type,
               epochs,
               config
           )
           
           if result and 'job_id' in result:
               st.info(f"Training job started: {result['job_id']}")
               
               # Create placeholders for live metrics
               if show_live_metrics:
                   metric_placeholder = st.empty()
                   chart_placeholder = st.empty()
                   
                   # Initialize metrics storage
                   training_metrics = {
                       'epochs': [],
                       'loss': [],
                       'val_loss': [],
                       'accuracy': []
                   }
               
               # Monitor training progress
               job_status = wait_for_job_completion(
                   result['job_id'], 
                   max_attempts=epochs * 10,
                   poll_interval=2
               )
               
               if job_status and job_status.get('status') == 'completed':
                   results = job_status.get('results', {})
                   
                   st.success("🎉 Model training completed!")
                   
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
                   
                   # Display final results
                   st.markdown("### Training Results")
                   
                   col1, col2, col3, col4 = st.columns(4)
                   with col1:
                       st.metric("Final Loss", f"{results.get('final_metrics', {}).get('loss', 0):.4f}")
                   with col2:
                       st.metric("Accuracy", f"{results.get('final_metrics', {}).get('accuracy', 0):.2%}")
                   with col3:
                       st.metric("Val Loss", f"{results.get('final_metrics', {}).get('validation_loss', 0):.4f}")
                   with col4:
                       st.metric("Training Time", f"{results.get('training_time', 0):.1f}s")
                   
                   # Model info
                   st.markdown("### Model Information")
                   st.markdown(f"""
                   <div class="info-box">
                       <b>Model ID:</b> {results.get('model_id', 'Unknown')}<br>
                       <b>Model Path:</b> {results.get('model_path', 'Unknown')}<br>
                       <b>Ready for inference:</b> ✅ Yes
                   </div>
                   """, unsafe_allow_html=True)
                   
                   # Refresh available models
                   models_response = api_client.get_models()
                   st.session_state.available_models = models_response.get('models', [])

def render_ml_generation():
   """ML Generation page"""
   st.title("🧪 ML-Based ODE Generation")
   st.markdown("Generate novel ODEs using trained machine learning models")
   
   # Check for trained models
   if not st.session_state.available_models:
       st.warning("No trained models available. Please train a model first.")
       if st.button("Go to ML Training"):
           st.session_state.current_page = "🤖 ML Training"
           st.rerun()
       return
   
   # Model selection
   st.markdown("### Select Model")
   
   model_options = []
   for model in st.session_state.available_models:
       model_desc = f"{model['name']} ({model.get('metadata', {}).get('model_type', 'Unknown')}) - Created: {model.get('created', 'Unknown')[:10]}"
       model_options.append((model['path'], model_desc, model))
   
   selected_model_index = st.selectbox(
       "Available Models",
       range(len(model_options)),
       format_func=lambda x: model_options[x][1]
   )
   
   selected_model_path = model_options[selected_model_index][0]
   selected_model_info = model_options[selected_model_index][2]
   
   # Display model info
   if selected_model_info:
       metadata = selected_model_info.get('metadata', {})
       st.markdown(f"""
       <div class="info-box">
           <b>Model Type:</b> {metadata.get('model_type', 'Unknown')}<br>
           <b>Trained on:</b> {metadata.get('dataset', 'Unknown')}<br>
           <b>Training Config:</b> {metadata.get('training_config', {}).get('epochs', 'Unknown')} epochs, 
           LR: {metadata.get('training_config', {}).get('learning_rate', 'Unknown')}
       </div>
       """, unsafe_allow_html=True)
   
   # Generation parameters
   st.markdown("### Generation Parameters")
   
   col1, col2 = st.columns(2)
   
   with col1:
       n_samples = st.slider(
           "Number of ODEs to Generate",
           min_value=5,
           max_value=100,
           value=20,
           step=5,
           help="Number of novel ODEs to generate"
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
       # Target specifications
       target_generator = st.selectbox(
           "Target Generator Style",
           ["Auto"] + st.session_state.available_generators,
           help="Specify a generator style or let the model choose"
       )
       
       target_function = st.selectbox(
           "Target Function Type",
           ["Auto"] + st.session_state.available_functions[:15],
           help="Specify a function type or let the model choose"
       )
   
   # Advanced options
   with st.expander("Advanced Generation Options"):
       col1, col2 = st.columns(2)
       
       with col1:
           complexity_min = st.number_input(
               "Min Complexity",
               min_value=10,
               max_value=500,
               value=50,
               step=10
           )
           complexity_max = st.number_input(
               "Max Complexity",
               min_value=complexity_min,
               max_value=1000,
               value=200,
               step=10
           )
       
       with col2:
           verify_generated = st.checkbox("Verify Generated ODEs", value=True)
           filter_duplicates = st.checkbox("Filter Duplicates", value=True)
   
   # Generate button
   if st.button("🎨 Generate Novel ODEs", type="primary"):
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
               job_status = wait_for_job_completion(result['job_id'], max_attempts=120)
               
               if job_status and job_status.get('status') == 'completed':
                   results = job_status.get('results', {})
                   generated_odes = results.get('odes', [])
                   
                   st.success(f"✨ Generated {len(generated_odes)} novel ODEs!")
                   
                   # Store results
                   st.session_state.ml_generated_odes.extend(generated_odes)
                   
                   # Display metrics
                   metrics = results.get('metrics', {})
                   if metrics:
                       col1, col2, col3 = st.columns(3)
                       
                       with col1:
                           st.metric("Total Generated", metrics.get('total_generated', len(generated_odes)))
                       with col2:
                           st.metric("Model Used", metadata.get('model_type', 'Unknown'))
                       with col3:
                           st.metric("Temperature", temperature)
                   
                   # Display generated ODEs
                   st.markdown("### Generated ODEs")
                   
                   # Options for display
                   display_options = st.columns([2, 1, 1])
                   with display_options[0]:
                       display_count = st.slider("Show ODEs", 1, len(generated_odes), min(10, len(generated_odes)))
                   with display_options[1]:
                       sort_by = st.selectbox("Sort by", ["Default", "Complexity", "Verified First"])
                   with display_options[2]:
                       show_plots = st.checkbox("Show plots", value=False)
                   
                   # Sort if needed
                   displayed_odes = generated_odes[:display_count]
                   if sort_by == "Complexity":
                       displayed_odes = sorted(displayed_odes, key=lambda x: x.get('complexity', 0), reverse=True)[:display_count]
                   elif sort_by == "Verified First":
                       displayed_odes = sorted(displayed_odes, key=lambda x: x.get('verified', False), reverse=True)[:display_count]
                   
                   for i, ode in enumerate(displayed_odes):
                       with st.expander(
                           f"Generated ODE {i+1}: {ode.get('generator', 'ML')} style, "
                           f"{'✅ Verified' if ode.get('verified') else '❓ Not Verified'}", 
                           expanded=(i < 3)
                       ):
                           render_ode_latex(
                               ode.get('ode', ''),
                               ode.get('solution', '') if ode.get('solution') else None,
                               "ML-Generated ODE"
                           )
                           
                           col1, col2, col3, col4 = st.columns(4)
                           with col1:
                               st.metric("Generator Style", ode.get('generator', 'Auto'))
                           with col2:
                               st.metric("Function Type", ode.get('function', 'Auto'))
                           with col3:
                               st.metric("Complexity", ode.get('complexity', 'N/A'))
                           with col4:
                               st.metric("Temperature", ode.get('temperature', temperature))
                           
                           if show_plots and ode.get('solution'):
                               fig = plot_solution(ode['solution'])
                               if fig:
                                   st.plotly_chart(fig, use_container_width=True)
                   
                   # Export options
                   st.markdown("### Export Generated ODEs")
                   col1, col2 = st.columns(2)
                   
                   with col1:
                       if st.button("📥 Download Generated ODEs"):
                           download_dataset(generated_odes, f"ml_generated_odes_{len(generated_odes)}.jsonl")
                   
                   with col2:
                       if st.button("💾 Save as Dataset"):
                           dataset_name = f"ml_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                           with st.spinner("Creating dataset..."):
                               result = api_client.create_dataset(generated_odes, dataset_name)
                               if result:
                                   st.success(f"Saved as dataset: {dataset_name}")
                                   # Refresh datasets
                                   datasets_response = api_client.list_datasets()
                                   st.session_state.available_datasets = datasets_response.get('datasets', [])

def render_analysis():
   """Analysis and visualization page"""
   st.title("📈 Analysis & Visualization")
   st.markdown("Analyze and visualize your ODE datasets")
   
   # Data source selection
   data_source = st.radio(
       "Select Data Source",
       ["Generated ODEs", "Batch Dataset", "ML Generated ODEs", "Load Dataset"],
       horizontal=True
   )
   
   # Load data based on selection
   odes_to_analyze = []
   
   if data_source == "Generated ODEs":
       odes_to_analyze = st.session_state.generated_odes
   elif data_source == "Batch Dataset":
       odes_to_analyze = st.session_state.batch_dataset
   elif data_source == "ML Generated ODEs":
       odes_to_analyze = st.session_state.ml_generated_odes
   elif data_source == "Load Dataset":
       if st.session_state.available_datasets:
           dataset_name = st.selectbox(
               "Select Dataset",
               [d['name'] for d in st.session_state.available_datasets]
           )
           # In a real implementation, you would load the dataset here
           st.info(f"Dataset '{dataset_name}' would be loaded for analysis")
       else:
           st.warning("No datasets available")
   
   if not odes_to_analyze:
       st.warning("No ODEs available for analysis. Generate some ODEs first.")
       return
   
   st.info(f"Analyzing {len(odes_to_analyze)} ODEs")
   
   # Convert to DataFrame for analysis
   df = pd.DataFrame(odes_to_analyze)
   
   # Analysis tabs
   tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Distributions", "Relationships", "Verification Analysis", "Export"])
   
   with tab1:
       st.markdown("### Dataset Overview")
       
       # Summary metrics
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           st.metric("Total ODEs", len(df))
       
       with col2:
           verified = df['verified'].sum() if 'verified' in df else 0
           st.metric("Verified", f"{verified} ({verified/len(df)*100:.1f}%)")
       
       with col3:
           unique_generators = df['generator'].nunique() if 'generator' in df else 0
           st.metric("Unique Generators", unique_generators)
       
       with col4:
           unique_functions = df['function'].nunique() if 'function' in df else 0
           st.metric("Unique Functions", unique_functions)
       
       # Generator distribution
       if 'generator' in df:
           st.markdown("### Generator Distribution")
           gen_counts = df['generator'].value_counts()
           
           fig = px.bar(
               x=gen_counts.index,
               y=gen_counts.values,
               labels={'x': 'Generator', 'y': 'Count'},
               title="ODEs by Generator",
               color=gen_counts.values,
               color_continuous_scale='viridis'
           )
           st.plotly_chart(fig, use_container_width=True)
       
       # Function distribution
       if 'function' in df:
           st.markdown("### Function Distribution")
           func_counts = df['function'].value_counts().head(15)
           
           fig = px.pie(
               values=func_counts.values,
               names=func_counts.index,
               title="Top 15 Functions Used"
           )
           st.plotly_chart(fig, use_container_width=True)
   
   with tab2:
       st.markdown("### Complexity Distribution")
       
       if 'complexity' in df:
           # Complexity histogram
           fig = px.histogram(
               df,
               x='complexity',
               nbins=50,
               title="Complexity Score Distribution",
               labels={'complexity': 'Complexity Score', 'count': 'Frequency'}
           )
           fig.update_layout(showlegend=False)
           st.plotly_chart(fig, use_container_width=True)
           
           # Complexity by generator
           if 'generator' in df:
               fig = px.box(
                   df,
                   x='generator',
                   y='complexity',
                   title="Complexity Distribution by Generator",
                   color='generator'
               )
               st.plotly_chart(fig, use_container_width=True)
       
       # Parameter distributions
       st.markdown("### Parameter Distributions")
       
       if 'parameters' in df.columns:
           # Extract parameters
           param_data = []
           for idx, row in df.iterrows():
               if isinstance(row.get('parameters'), dict):
                   params = row['parameters']
                   param_data.append({
                       'alpha': params.get('alpha', 0),
                       'beta': params.get('beta', 0),
                       'M': params.get('M', 0)
                   })
           
           if param_data:
               param_df = pd.DataFrame(param_data)
               
               # Create parameter distribution plots
               fig = go.Figure()
               
               for param in ['alpha', 'beta', 'M']:
                   if param in param_df:
                       fig.add_trace(go.Violin(
                           y=param_df[param],
                           name=param,
                           box_visible=True,
                           meanline_visible=True
                       ))
               
               fig.update_layout(
                   title="Parameter Value Distributions",
                   yaxis_title="Value",
                   showlegend=True
               )
               st.plotly_chart(fig, use_container_width=True)
   
   with tab3:
       st.markdown("### Relationship Analysis")
       
       # Verification rate by generator
       if 'generator' in df and 'verified' in df:
           ver_by_gen = df.groupby('generator')['verified'].agg(['mean', 'count'])
           ver_by_gen = ver_by_gen.reset_index()
           ver_by_gen['mean'] = ver_by_gen['mean'] * 100
           
           fig = px.bar(
               ver_by_gen,
               x='generator',
               y='mean',
               text='count',
               title="Verification Rate by Generator",
               labels={'mean': 'Verification Rate (%)', 'generator': 'Generator'},
               color='mean',
               color_continuous_scale='RdYlGn'
           )
           fig.update_traces(texttemplate='%{text} ODEs', textposition='outside')
           st.plotly_chart(fig, use_container_width=True)
       
       # Complexity vs Verification
       if 'complexity' in df and 'verified' in df:
           # Create bins for complexity
           df['complexity_bin'] = pd.qcut(df['complexity'], q=10, labels=[f"D{i}" for i in range(1, 11)])
           ver_by_complexity = df.groupby('complexity_bin')['verified'].mean() * 100
           
           fig = px.line(
               x=ver_by_complexity.index,
               y=ver_by_complexity.values,
               title="Verification Rate by Complexity Decile",
               labels={'x': 'Complexity Decile', 'y': 'Verification Rate (%)'},
               markers=True
           )
           st.plotly_chart(fig, use_container_width=True)
       
       # Generator-Function heatmap
       if 'generator' in df and 'function' in df:
           # Create cross-tabulation
           cross_tab = pd.crosstab(df['generator'], df['function'])
           
           # Limit to top functions for readability
           top_functions = df['function'].value_counts().head(10).index
           cross_tab = cross_tab[cross_tab.columns.intersection(top_functions)]
           
           fig = px.imshow(
               cross_tab,
               title="Generator-Function Usage Heatmap",
               labels=dict(x="Function", y="Generator", color="Count"),
               aspect="auto",
               color_continuous_scale='YlOrRd'
           )
           st.plotly_chart(fig, use_container_width=True)
   
   with tab4:
       st.markdown("### Verification Analysis")
       
       if 'verified' in df:
           # Verification methods
           if 'properties' in df.columns:
               verification_methods = []
               for idx, row in df.iterrows():
                   if isinstance(row.get('properties'), dict):
                       method = row['properties'].get('verification_method', 'Unknown')
                       verification_methods.append(method)
               
               if verification_methods:
                   method_counts = pd.Series(verification_methods).value_counts()
                   
                   fig = px.pie(
                       values=method_counts.values,
                       names=method_counts.index,
                       title="Verification Methods Used",
                       hole=0.4
                   )
                   st.plotly_chart(fig, use_container_width=True)
           
           # Verification confidence distribution
           if 'properties' in df.columns:
               confidences = []
               for idx, row in df.iterrows():
                   if isinstance(row.get('properties'), dict):
                       conf = row['properties'].get('verification_confidence', 0)
                       confidences.append(conf)
               
               if confidences:
                   fig = px.histogram(
                       x=confidences,
                       nbins=20,
                       title="Verification Confidence Distribution",
                       labels={'x': 'Confidence', 'count': 'Frequency'}
                   )
                   st.plotly_chart(fig, use_container_width=True)
       
       # Failed verifications analysis
       if 'verified' in df:
           failed_df = df[~df['verified']]
           if len(failed_df) > 0:
               st.markdown(f"### Failed Verifications ({len(failed_df)} ODEs)")
               
               # Reasons for failure
               if 'generator' in failed_df:
                   failed_by_gen = failed_df['generator'].value_counts()
                   
                   fig = px.bar(
                       x=failed_by_gen.index,
                       y=failed_by_gen.values,
                       title="Failed Verifications by Generator",
                       labels={'x': 'Generator', 'y': 'Count'},
                       color=failed_by_gen.values,
                       color_continuous_scale='Reds'
                   )
                   st.plotly_chart(fig, use_container_width=True)
   
   with tab5:
       st.markdown("### Export Analysis Results")
       
       # Analysis summary
       analysis_summary = {
           "total_odes": len(df),
           "verified": int(df['verified'].sum()) if 'verified' in df else 0,
           "verification_rate": float(df['verified'].mean()) if 'verified' in df else 0,
           "unique_generators": int(df['generator'].nunique()) if 'generator' in df else 0,
           "unique_functions": int(df['function'].nunique()) if 'function' in df else 0,
           "avg_complexity": float(df['complexity'].mean()) if 'complexity' in df else 0,
           "std_complexity": float(df['complexity'].std()) if 'complexity' in df else 0
       }
       
       # Display summary
       st.json(analysis_summary)
       
       # Export options
       col1, col2, col3 = st.columns(3)
       
       with col1:
           if st.button("📊 Export Analysis Report"):
               export_analysis_report(analysis_summary, odes_to_analyze)
       
       with col2:
           if st.button("📈 Export Charts as HTML"):
               st.info("Chart export functionality would be implemented here")
       
       with col3:
           if st.button("📥 Export Filtered Dataset"):
               # Allow filtering before export
               if 'verified' in df:
                   export_verified_only = st.checkbox("Export verified ODEs only")
                   if export_verified_only:
                       filtered_odes = [ode for ode in odes_to_analyze if ode.get('verified', False)]
                       download_dataset(filtered_odes, "filtered_verified_odes.jsonl")
                   else:
                       download_dataset(odes_to_analyze, "analyzed_odes.jsonl")

def render_system_tools():
   """System tools and utilities"""
   st.title("🛠️ System Tools")
   st.markdown("System utilities and maintenance tools")
   
   tab1, tab2, tab3, tab4 = st.tabs(["API Status", "Metrics", "Configuration", "Utilities"])
   
   with tab1:
       st.markdown("### API Status & Health")
       
       # Refresh status
       if st.button("🔄 Refresh Status"):
           asyncio.run(check_api_status())
           st.rerun()
       
       # Display detailed status
       if st.session_state.api_status:
           status = st.session_state.api_status
           
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown("**System Status**")
               status_color = "#10b981" if status.get('status') == 'healthy' else "#ef4444"
               st.markdown(f"""
               <div class="info-box" style="border-left: 4px solid {status_color};">
                   <b>Status:</b> {status.get('status', 'Unknown')}<br>
                   <b>Timestamp:</b> {status.get('timestamp', 'Unknown')}<br>
                   <b>Redis:</b> {status.get('redis', 'Unknown')}<br>
                   <b>ML Enabled:</b> {'✅ Yes' if status.get('ml_enabled') else '❌ No'}
               </div>
               """, unsafe_allow_html=True)
           
           with col2:
               st.markdown("**Available Resources**")
               st.markdown(f"""
               <div class="info-box">
                   <b>Generators:</b> {status.get('generators', 0)}<br>
                   <b>Functions:</b> {status.get('functions', 0)}<br>
                   <b>Active Jobs:</b> {stats.get('active_jobs', 0) if 'stats' in locals() else 'N/A'}<br>
                   <b>API Version:</b> v2.0.0
               </div>
               """, unsafe_allow_html=True)
       
       # API Statistics
       st.markdown("### API Statistics")
       stats = api_client.get_statistics()
       
       if stats:
           # Create metrics dashboard
           col1, col2, col3 = st.columns(3)
           
           with col1:
               st.metric(
                   "24h Generated",
                   f"{stats.get('total_generated_24h', 0):,}",
                   help="Total ODEs generated in last 24 hours"
               )
           
           with col2:
               st.metric(
                   "Verification Rate",
                   f"{stats.get('verification_success_rate', 0):.1%}",
                   help="Overall verification success rate"
               )
           
           with col3:
               st.metric(
                   "Active Jobs",
                   stats.get('active_jobs', 0),
                   help="Currently processing jobs"
               )
   
   with tab2:
       st.markdown("### System Metrics")
       st.markdown("Prometheus metrics for system monitoring")
       
       if st.button("📊 Fetch Current Metrics"):
           metrics_text = api_client.get_metrics()
           
           if metrics_text:
               # Parse and display key metrics
               st.text_area(
                   "Raw Prometheus Metrics",
                   metrics_text[:2000] + "..." if len(metrics_text) > 2000 else metrics_text,
                   height=300
               )
               
               # Parse some key metrics
               if "ode_generation_total" in metrics_text:
                   st.success("✅ Generation metrics available")
               if "api_request_duration_seconds" in metrics_text:
                   st.success("✅ API performance metrics available")
               if "active_jobs" in metrics_text:
                   st.success("✅ Job tracking metrics available")
   
   with tab3:
       st.markdown("### Configuration")
       
       # Display current configuration
       st.markdown("**Current API Configuration**")
       config_info = {
           "API URL": API_BASE_URL,
           "API Key": f"{API_KEY[:8]}..." if len(API_KEY) > 8 else API_KEY,
           "Timeout": "30 seconds",
           "Max Retries": "3"
       }
       
       for key, value in config_info.items():
           st.text(f"{key}: {value}")
       
       # Configuration editor
       with st.expander("Edit Configuration"):
           new_api_url = st.text_input("API URL", value=API_BASE_URL)
           new_api_key = st.text_input("API Key", value=API_KEY, type="password")
           
           if st.button("Update Configuration"):
               st.warning("Configuration update would be implemented here")
   
   with tab4:
       st.markdown("### Utilities")
       
       # Cache management
       st.markdown("**Cache Management**")
       col1, col2 = st.columns(2)
       
       with col1:
           if st.button("🗑️ Clear Generated ODEs"):
               st.session_state.generated_odes = []
               st.session_state.batch_dataset = []
               st.session_state.ml_generated_odes = []
               st.success("Cleared all cached ODEs")
       
       with col2:
           if st.button("🔄 Reset Session"):
               for key in list(st.session_state.keys()):
                   del st.session_state[key]
               st.success("Session reset. Please refresh the page.")
       
       # Data export
       st.markdown("**Data Export**")
       
       if st.button("📦 Export All Data"):
           # Create a comprehensive export
           export_data = {
               "timestamp": datetime.now().isoformat(),
               "generated_odes": st.session_state.generated_odes,
               "batch_dataset": st.session_state.batch_dataset,
               "ml_generated_odes": st.session_state.ml_generated_odes,
               "datasets": st.session_state.available_datasets,
               "models": st.session_state.available_models
           }
           
           # Create zip file with all data
           st.info("Complete data export would be created here")
       
       # System diagnostics
       st.markdown("**System Diagnostics**")
       
       if st.button("🔍 Run Diagnostics"):
           with st.spinner("Running system diagnostics..."):
               diagnostics = {
                   "api_connection": "✅ Connected" if st.session_state.api_status else "❌ Disconnected",
                   "cached_odes": len(st.session_state.generated_odes) + len(st.session_state.batch_dataset) + len(st.session_state.ml_generated_odes),
                   "available_datasets": len(st.session_state.available_datasets),
                   "available_models": len(st.session_state.available_models),
                   "session_size": sum(len(str(v)) for v in st.session_state.values()) / 1024 / 1024  # MB
               }
               
               st.json(diagnostics)

def render_documentation():
   """Documentation page"""
   st.title("📚 Documentation")
   st.markdown("Complete guide to the ODE Master Generator System")
   
   # Documentation sections
   doc_section = st.selectbox(
       "Select Documentation Section",
       ["Quick Start", "Generators", "Functions", "API Reference", "ML Models", "Examples", "Troubleshooting"]
   )
   
   if doc_section == "Quick Start":
       st.markdown("""
       ## Quick Start Guide
       
       Welcome to the ODE Master Generator System! This platform allows you to:
       
       1. **Generate ODEs** - Create ordinary differential equations with exact solutions
       2. **Verify Solutions** - Validate ODE solutions using multiple methods
       3. **Create Datasets** - Build large datasets for analysis and ML
       4. **Train ML Models** - Train neural networks to learn ODE patterns
       5. **Generate Novel ODEs** - Use ML to create new equations
       
       ### Getting Started
       
       1. **Check System Status** - Ensure the API is online (green indicator in sidebar)
       2. **Quick Generate** - Try generating a single ODE to familiarize yourself
       3. **Batch Generation** - Create larger datasets for analysis
       4. **ML Workflow** - Train models and generate novel ODEs
       
       ### Basic Workflow
       
       ```
       1. Generate ODEs (Quick or Batch)
          ↓
       2. Verify Solutions
          ↓
       3. Create Dataset
          ↓
       4. Train ML Model
          ↓
       5. Generate Novel ODEs
       ```
       """)
   
   elif doc_section == "Generators":
       st.markdown("""
       ## ODE Generators
       
       The system includes both linear and nonlinear generators:
       
       ### Linear Generators
       
       - **L1**: y''(x) + y(x) = RHS
       - **L2**: y''(x) + y'(x) = RHS  
       - **L3**: y(x) + y'(x) = RHS
       - **L4**: Pantograph equation with y(x/a) term
       
       ### Nonlinear Generators
       
       - **N1**: (y''(x))^q + y(x) = RHS
       - **N2**: (y''(x))^q + (y'(x))^v = RHS
       - **N3**: y(x) + (y'(x))^v = RHS
       - **N4**: exp(y''(x)) + y(x) = RHS
       - **N5**: sin(y''(x)) + cos(y'(x)) + y(x) = RHS
       - **N6**: Nonlinear pantograph equation
       - **N7**: Composite nonlinearity with customizable functions
       
       ### Parameters
       
       - **α (alpha)**: Shift parameter in function argument
       - **β (beta)**: Scale parameter in exponential term
       - **M**: Constant term in solution
       - **q, v**: Powers for nonlinear terms
       - **a**: Delay factor for pantograph equations
       """)
   
   elif doc_section == "Functions":
       st.markdown("""
       ## Mathematical Functions
       
       The system supports a wide variety of mathematical functions:
       
       ### Categories
       
       **Polynomial Functions**
       - identity, quadratic, cubic, quartic, quintic
       
       **Exponential Functions**
       - exponential, exp_scaled, exp_quadratic, exp_negative
       
       **Trigonometric Functions**
       - sine, cosine, tangent_safe, sine_scaled, cosine_scaled
       
       **Hyperbolic Functions**
       - sinh, cosh, tanh
       
       **Logarithmic Functions**
       - log_safe, log_shifted, log_scaled
       
       **Rational Functions**
       - rational_simple, rational_stable, rational_cubic
       
       **Composite Functions**
       - exp_sin, sin_exp, gaussian, bessel_like
       """)
   
   elif doc_section == "API Reference":
       st.markdown("""
       ## API Reference
       
       ### Endpoints
       
       #### POST /api/v1/generate
       Generate ODEs with specified parameters
       
       ```json
       {
           "generator": "L1",
           "function": "sine",
           "parameters": {
               "alpha": 1.0,
               "beta": 1.0,
               "M": 0.0
           },
           "count": 1,
           "verify": true
       }
       ```
       
       #### POST /api/v1/batch_generate
       Generate multiple ODEs in batch
       
       #### POST /api/v1/verify
       Verify an ODE solution
       
       #### POST /api/v1/ml/train
       Train ML model on dataset
       
       #### POST /api/v1/ml/generate
       Generate ODEs using trained model
       
       ### Authentication
       
       All API requests require the `X-API-Key` header.
       """)
   
   elif doc_section == "ML Models":
       st.markdown("""
       ## Machine Learning Models
       
       ### Available Models
       
       **PatternNet**
       - Fast pattern recognition network
       - Good for verification prediction
       - Lightweight and efficient
       
       **Transformer**
       - Advanced sequence modeling
       - Better for complex patterns
       - Requires more training data
       
       **VAE (Variational Autoencoder)**
       - Generative model
       - Good for creating novel ODEs
       - Learns latent representations
       
       ### Training Tips
       
       1. **Dataset Size**: Aim for at least 1000 ODEs
       2. **Diversity**: Include various generators and functions
       3. **Balance**: Ensure mix of verified/unverified ODEs
       4. **Epochs**: Start with 50, increase if needed
       5. **Learning Rate**: Default 0.001 works well
       """)
   
   elif doc_section == "Examples":
       st.markdown("""
       ## Examples
       
       ### Example 1: Simple Linear ODE
       
       Generator: L1  
       Function: sine  
       Parameters: α=1, β=1, M=0
       
       Result:
       ```
       ODE: y''(x) + y(x) = π·sin(1 + e^(-x))
       Solution: y(x) = π·(sin(1 + e^(-x)) - sin(1))
       ```
       
       ### Example 2: Nonlinear ODE
       
       Generator: N1  
       Function: exponential  
       Parameters: α=0, β=2, M=0.5, q=2
       
       Result:
       ```
       ODE: (y''(x))² + y(x) = ...
       Solution: y(x) = ...
       ```
       
       ### Example 3: Complete Workflow
       
       ```python
       # 1. Generate batch dataset
       generators = ["L1", "L2", "N1", "N2"]
       functions = ["sine", "cosine", "exponential"]
       samples = 10
       
       # 2. Create dataset
       dataset_name = "training_set"
       
       # 3. Train model
       model_type = "pattern_net"
       epochs = 50
       
       # 4. Generate novel ODEs
       n_samples = 20
       temperature = 0.8
       ```
       """)
   
   elif doc_section == "Troubleshooting":
       st.markdown("""
       ## Troubleshooting
       
       ### Common Issues
       
       **API Connection Failed**
       - Check your internet connection
       - Verify API URL is correct
       - Ensure API key is valid
       
       **Generation Failed**
       - Try different parameter values
       - Check generator/function compatibility
       - Reduce batch size if timeout occurs
       
       **Verification Failed**
       - Some complex ODEs may not verify
       - Try different verification methods
       - Check solution format
       
       **ML Training Issues**
       - Ensure dataset has sufficient samples
       - Check if ML features are enabled
       - Try reducing batch size or learning rate
       
       ### Performance Tips
       
       1. **Batch Generation**: Use smaller batches for complex generators
       2. **Parallel Processing**: Increase workers for faster generation
       3. **Caching**: Clear cache regularly to free memory
       4. **Dataset Size**: Start small and scale up
       
       ### Getting Help
       
       - Check the API logs for detailed errors
       - Use the System Tools diagnostics
       - Contact support with job IDs for failed operations
       """)

# Run the app
if __name__ == "__main__":
   main()
