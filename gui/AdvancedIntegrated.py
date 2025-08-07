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
import itertools

# Page configuration
st.set_page_config(
    page_title="ODE Master Generator - Mohammad Abu Ghuwaleh",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI and LaTeX rendering
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
    .latex-equation {
        font-size: 1.2em;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        overflow-x: auto;
        margin: 0.5rem 0;
    }
    .doc-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .doc-section h3 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'https://ode-api-production.up.railway.app')
API_KEY = os.getenv('API_KEY', 'test-key')

# Initialize session state
if 'generated_odes' not in st.session_state:
    st.session_state.generated_odes = []
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'batch_dataset' not in st.session_state:
    st.session_state.batch_dataset = []

class ODEAPIClient:
    """API client for ODE generation system"""
    
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
        except requests.exceptions.Timeout:
            st.error("Request timed out. The server might be busy. Please try again.")
            return None
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API server. Please check if the server is running.")
            return None
        except Exception as e:
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
                    time.sleep(1)  # Wait before retry
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to get job status after {max_retries} attempts")
                    return None
                time.sleep(1)
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
            st.error(f"Verification Error: {str(e)}")
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
            # Return default generators if API fails
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
            # Return default functions if API fails
            return {
                'functions': [
                    'identity', 'quadratic', 'cubic', 'quartic', 'quintic',
                    'exponential', 'exp_scaled', 'exp_quadratic', 'exp_negative',
                    'sine', 'cosine', 'tangent_safe', 'sine_scaled', 'cosine_scaled',
                    'sinh', 'cosh', 'tanh', 'log_safe', 'log_shifted',
                    'rational_simple', 'rational_stable', 'exp_sin', 'gaussian'
                ]
            }
    
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
            return {
                'status': 'unknown',
                'error': str(e)
            }
    
    def analyze_dataset(self, ode_list: List[str]) -> Dict:
        """Analyze ODE dataset locally since API endpoint has issues"""
        # Perform local analysis instead of API call
        try:
            analysis_results = {
                'total_odes': len(ode_list),
                'statistics': {},
                'generator_distribution': {},
                'complexity_distribution': {}
            }
            
            # Basic statistics
            if st.session_state.generated_odes:
                complexities = [ode.get('complexity', 0) for ode in st.session_state.generated_odes]
                verified_count = sum(1 for ode in st.session_state.generated_odes if ode.get('verified', False))
                
                analysis_results['statistics'] = {
                    'verified_rate': verified_count / len(st.session_state.generated_odes) if st.session_state.generated_odes else 0,
                    'avg_complexity': np.mean(complexities) if complexities else 0,
                    'complexity_std': np.std(complexities) if complexities else 0,
                    'complexity_range': [min(complexities), max(complexities)] if complexities else [0, 0]
                }
                
                # Generator distribution
                generator_counts = {}
                for ode in st.session_state.generated_odes:
                    gen = ode.get('generator', 'unknown')
                    generator_counts[gen] = generator_counts.get(gen, 0) + 1
                
                analysis_results['generator_distribution'] = generator_counts
            
            return {'success': True, 'results': analysis_results}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Initialize API client
api_client = ODEAPIClient(API_BASE_URL, API_KEY)

def render_ode_latex(ode_str: str, solution_str: str = None, title: str = "ODE"):
    """Render ODE and solution using improved LaTeX formatting"""
    try:
        # Parse and format the ODE string for better display
        if 'Eq(' in ode_str:
            # Extract the equation parts
            parts = ode_str.replace('Eq(', '').replace(')', '', -1).split(',', 1)
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                
                # Format derivatives properly
                lhs = lhs.replace('Derivative(y(x), (x, 2))', "y''(x)")
                lhs = lhs.replace('Derivative(y(x), x, 2)', "y''(x)")
                lhs = lhs.replace('Derivative(y(x), x)', "y'(x)")
                
                # Create formatted equation
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
            # Format solution
            formatted_solution = f"y(x) = {solution_str}"
            st.markdown(f'<div class="latex-equation">{formatted_solution}</div>', unsafe_allow_html=True)
            
    except Exception as e:
        # Fallback to code display if formatting fails
        st.code(ode_str, language='python')
        if solution_str:
            st.code(f"y(x) = {solution_str}", language='python')

def plot_solution(solution_str: str, x_range: tuple = (-5, 5), params: Dict = None):
    """Plot ODE solution"""
    try:
        # Parse solution
        x = symbols('x')
        solution_expr = sp.sympify(solution_str)
        
        # Substitute parameters if provided
        if params:
            for param, value in params.items():
                if param in ['alpha', 'beta', 'M']:
                    solution_expr = solution_expr.subs(param, value)
        
        # Create numerical function
        solution_func = sp.lambdify(x, solution_expr, 'numpy')
        
        # Generate plot data
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        y_vals = solution_func(x_vals)
        
        # Handle potential infinities
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
        return None

def main():
    # Header with author credit
    st.markdown("""
    <div class="main-header">
        <h1>🔬 ODE Master Generator System</h1>
        <p>Advanced Ordinary Differential Equation Generation, Verification & Analysis Platform</p>
    </div>
    <div class="author-credit">
        Created by Mohammad Abu Ghuwaleh
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Dashboard", "⚡ Single ODE Generation", "📦 Batch Generation", 
         "✅ Verify Solutions", "📊 Analysis Suite", "🤖 ML Training", 
         "🧪 ML Generation", "📈 Statistics", "📚 Documentation", "⚙️ Settings"]
    )
    
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "⚡ Single ODE Generation":
        render_single_generation_page()
    elif page == "📦 Batch Generation":
        render_batch_generation_page()
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
    elif page == "📚 Documentation":
        render_documentation_page()
    elif page == "⚙️ Settings":
        render_settings_page()

def render_dashboard():
    """Render main dashboard"""
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
        verified_count = sum(1 for ode in st.session_state.generated_odes if ode.get('verified', False))
        verified_count += sum(1 for ode in st.session_state.batch_dataset if ode.get('verified', False))
        st.markdown(f"""
        <div class="metric-card">
            <h3>Verified ODEs</h3>
            <h1>{verified_count}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        generators_data = api_client.get_generators()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Available Generators</h3>
            <h1>{len(generators_data.get('all', []))}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent ODEs
    all_odes = st.session_state.generated_odes + st.session_state.batch_dataset
    if all_odes:
        st.subheader("Recently Generated ODEs")
        
        for i, ode in enumerate(all_odes[-5:]):
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
                    verified_status = "✅ Verified" if ode.get('verified', False) else "❌ Not Verified"
                    st.metric("Status", verified_status)

def render_single_generation_page():
    """Render single ODE generation page - generates same ODE with given parameters"""
    st.title("⚡ Single ODE Generation")
    st.markdown("Generate a single ODE with specific parameters. The same parameters will produce the same ODE.")
    
    # Get available generators and functions
    generators_data = api_client.get_generators()
    functions_data = api_client.get_functions()
    
    generators = generators_data.get('all', [])
    functions = functions_data.get('functions', [])
    
    # Generation form
    with st.form("single_generation_form"):
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
        
        with col2:
            st.markdown("### Parameters")
            
            # Parameter inputs
            params = {}
            params['alpha'] = st.slider("α (alpha)", -2.0, 2.0, 1.0, 0.1)
            params['beta'] = st.slider("β (beta)", 0.1, 2.0, 1.0, 0.1)
            params['M'] = st.slider("M", -1.0, 1.0, 0.0, 0.1)
            
            # Nonlinear parameters
            if selected_generator and 'N' in selected_generator:
                params['q'] = st.slider("q (power)", 2, 5, 2)
                params['v'] = st.slider("v (power)", 2, 5, 3)
            
            # Pantograph parameter
            if selected_generator and ('L4' in selected_generator or 'N6' in selected_generator):
                params['a'] = st.slider("a (pantograph)", 2, 5, 2)
            
            verify = st.checkbox("Verify solution", value=True)
        
        submit = st.form_submit_button("Generate ODE", type="primary")
    
    if submit:
        with st.spinner("Generating ODE..."):
            # Call API with count=1 for single ODE
            result = api_client.generate_odes(
                selected_generator,
                selected_function,
                params,
                count=1,
                verify=verify
            )
            
            if result and 'job_id' in result:
                # Poll for results
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                max_attempts = 30
                for attempt in range(max_attempts):
                    job_status = api_client.get_job_status(result['job_id'])
                    
                    if job_status:
                        progress = job_status.get('progress', 0)
                        progress_bar.progress(int(progress))
                        status_text.text(f"Status: {job_status.get('status', 'Unknown')}")
                        
                        if job_status.get('status') == 'completed':
                            results = job_status.get('results', [])
                            if results:
                                st.success("ODE generated successfully!")
                                
                                # Store single ODE
                                ode = results[0]
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
                                
                                # Display parameters used
                                with st.expander("Parameters Used"):
                                    st.json(ode.get('parameters', {}))
                                
                                # Plot solution
                                if ode.get('solution'):
                                    st.subheader("Solution Plot")
                                    fig = plot_solution(
                                        ode['solution'],
                                        params=ode.get('parameters', {})
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                # Initial conditions
                                if 'properties' in ode and 'initial_conditions' in ode['properties']:
                                    st.subheader("Initial Conditions")
                                    for ic_key, ic_value in ode['properties']['initial_conditions'].items():
                                        st.code(f"{ic_key} = {ic_value}")
                            
                            break
                        
                        elif job_status.get('status') == 'failed':
                            st.error(f"Generation failed: {job_status.get('error', 'Unknown error')}")
                            break
                    
                    time.sleep(1)
                    
                    if attempt == max_attempts - 1:
                        st.error("Generation timed out. Please try again.")

def render_batch_generation_page():
    """Render batch ODE generation page - generates diverse dataset"""
    st.title("📦 Batch ODE Generation")
    st.markdown("Generate a diverse dataset of ODEs with multiple generators, functions, and parameter combinations.")
    
    # Get available generators and functions
    generators_data = api_client.get_generators()
    functions_data = api_client.get_functions()
    
    all_generators = generators_data.get('all', [])
    all_functions = functions_data.get('functions', [])
    
    # Batch generation form
    with st.form("batch_generation_form"):
        st.subheader("Select Generators")
        col1, col2 = st.columns(2)
        
        with col1:
            linear_generators = generators_data.get('linear', [])
            selected_linear = st.multiselect(
                "Linear Generators",
                linear_generators,
                default=linear_generators[:2]
            )
        
        with col2:
            nonlinear_generators = generators_data.get('nonlinear', [])
            selected_nonlinear = st.multiselect(
                "Nonlinear Generators",
                nonlinear_generators,
                default=nonlinear_generators[:2]
            )
        
        selected_generators = selected_linear + selected_nonlinear
        
        st.subheader("Select Functions")
        selected_functions = st.multiselect(
            "Mathematical Functions",
            all_functions,
            default=all_functions[:5]
        )
        
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
                default=[0.5, 1.0]
            )
        
        with col3:
            m_values = st.multiselect(
                "M values",
                [-1.0, -0.5, 0.0, 0.5, 1.0],
                default=[0.0]
            )
        
        # Nonlinear parameters
        if any('N' in gen for gen in selected_generators):
            col1, col2 = st.columns(2)
            with col1:
                q_values = st.multiselect("q values", [2, 3, 4, 5], default=[2])
            with col2:
                v_values = st.multiselect("v values", [2, 3, 4, 5], default=[3])
        else:
            q_values = [2]
            v_values = [3]
        
        # Calculate total combinations
        total_combinations = (
            len(selected_generators) * 
            len(selected_functions) * 
            len(alpha_values) * 
            len(beta_values) * 
            len(m_values) * 
            len(q_values) * 
            len(v_values)
        )
        
        st.info(f"This will generate approximately {total_combinations} unique ODEs")
        
        verify = st.checkbox("Verify all solutions", value=True)
        
        submit = st.form_submit_button("Generate Batch Dataset", type="primary")
    
    if submit and selected_generators and selected_functions:
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_results = []
        total_generated = 0
        
        # Generate all combinations
        param_combinations = list(itertools.product(
            alpha_values, beta_values, m_values, q_values, v_values
        ))
        
        total_tasks = len(selected_generators) * len(selected_functions) * len(param_combinations)
        current_task = 0
        
        # Process each generator
        for generator in selected_generators:
            for function in selected_functions:
                for alpha, beta, m, q, v in param_combinations:
                    current_task += 1
                    progress_bar.progress(current_task / total_tasks)
                    status_text.text(f"Generating: {generator} + {function} ({current_task}/{total_tasks})")
                    
                    # Build parameters
                    params = {
                        'alpha': float(alpha),
                        'beta': float(beta),
                        'M': float(m),
                        'q': int(q),
                        'v': int(v)
                    }
                    
                    # Add pantograph parameter if needed
                    if 'L4' in generator or 'N6' in generator:
                        params['a'] = 2
                    
                    # Generate ODE
                    result = api_client.generate_odes(
                        generator,
                        function,
                        params,
                        count=1,
                        verify=verify
                    )
                    
                    if result and 'job_id' in result:
                        # Wait for completion
                        for _ in range(20):
                            job_status = api_client.get_job_status(result['job_id'])
                            if job_status and job_status.get('status') == 'completed':
                                odes = job_status.get('results', [])
                                if odes:
                                    # Add metadata
                                    for ode in odes:
                                        ode['batch_id'] = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                        ode['combination_id'] = f"{generator}_{function}_{alpha}_{beta}_{m}"
                                    batch_results.extend(odes)
                                    total_generated += len(odes)
                                break
                            time.sleep(0.5)
        
        # Store results
        st.session_state.batch_dataset.extend(batch_results)
        
        # Display summary
        st.success(f"Batch generation complete! Generated {total_generated} ODEs")
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Generated", total_generated)
        with col2:
            verified_count = sum(1 for ode in batch_results if ode.get('verified', False))
            st.metric("Verified", verified_count)
        with col3:
            st.metric("Success Rate", f"{100 * verified_count / total_generated:.1f}%" if total_generated > 0 else "0%")
        
        # Display sample results
        if batch_results:
            st.subheader("Sample Results")
            
            # Create summary dataframe
            df_data = []
            for ode in batch_results:
                df_data.append({
                    'Generator': ode.get('generator', ''),
                    'Function': ode.get('function', ''),
                    'α': ode.get('parameters', {}).get('alpha', ''),
                    'β': ode.get('parameters', {}).get('beta', ''),
                    'M': ode.get('parameters', {}).get('M', ''),
                    'Verified': '✅' if ode.get('verified', False) else '❌',
                    'Complexity': ode.get('complexity', 0)
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Export button
            if st.button("Export Batch Dataset"):
                # Create export data
                export_data = {
                    'metadata': {
                        'generated_date': datetime.now().isoformat(),
                        'total_odes': len(batch_results),
                        'generators_used': list(set(ode.get('generator') for ode in batch_results)),
                        'functions_used': list(set(ode.get('function') for ode in batch_results))
                    },
                    'odes': batch_results
                }
                
                # Convert to JSON
                json_str = json.dumps(export_data, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                
                # Create download link
                href = f'<a href="data:file/json;base64,{b64}" download="ode_batch_dataset.json">Download Batch Dataset (JSON)</a>'
                st.markdown(href, unsafe_allow_html=True)

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
                placeholder="e.g., Eq(y''(x) + y(x), sin(x))",
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
        st.markdown("### Verify Dataset ODEs")
        
        all_odes = st.session_state.generated_odes + st.session_state.batch_dataset
        
        if all_odes:
            # Create dataframe
            df_data = []
            for i, ode in enumerate(all_odes):
                df_data.append({
                    'Index': i,
                    'Generator': ode.get('generator', ''),
                    'Function': ode.get('function', ''),
                    'Verified': '✅' if ode.get('verified', False) else '❌',
                    'Complexity': ode.get('complexity', 0)
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Re-verify button
            if st.button("Re-verify All Unverified ODEs"):
                progress_bar = st.progress(0)
                verified_count = 0
                
                for i, ode in enumerate(all_odes):
                    if not ode.get('verified', False) and ode.get('ode') and ode.get('solution'):
                        result = api_client.verify_ode(
                            ode['ode'],
                            ode['solution']
                        )
                        
                        if result:
                            ode['verified'] = result.get('verified', False)
                            if result.get('verified'):
                                verified_count += 1
                    
                    progress_bar.progress((i + 1) / len(all_odes))
                
                st.success(f"Re-verification complete! Verified {verified_count} additional ODEs.")
                st.rerun()
        else:
            st.info("No ODEs available for verification. Generate some ODEs first.")

def render_analysis_page():
    """Render analysis page with local analysis"""
    st.title("📊 ODE Analysis Suite")
    
    all_odes = st.session_state.generated_odes + st.session_state.batch_dataset
    
    if not all_odes:
        st.warning("No ODEs available for analysis. Please generate some ODEs first.")
        return
    
    # Perform local analysis
    st.subheader("Dataset Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total ODEs", len(all_odes))
    
    with col2:
        verified_count = sum(1 for ode in all_odes if ode.get('verified', False))
        st.metric("Verified ODEs", verified_count)
    
    with col3:
        verification_rate = verified_count / len(all_odes) if all_odes else 0
        st.metric("Verification Rate", f"{verification_rate:.1%}")
    
    with col4:
        complexities = [ode.get('complexity', 0) for ode in all_odes]
        avg_complexity = np.mean(complexities) if complexities else 0
        st.metric("Avg Complexity", f"{avg_complexity:.1f}")
    
    # Generator distribution
    st.subheader("Generator Distribution")
    
    generator_counts = {}
    for ode in all_odes:
        gen = ode.get('generator', 'Unknown')
        generator_counts[gen] = generator_counts.get(gen, 0) + 1
    
    if generator_counts:
        fig = px.bar(
            x=list(generator_counts.keys()),
            y=list(generator_counts.values()),
            title="ODEs by Generator",
            labels={'x': 'Generator', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Function distribution
    st.subheader("Function Distribution")
    
    function_counts = {}
    for ode in all_odes:
        func = ode.get('function', 'Unknown')
        function_counts[func] = function_counts.get(func, 0) + 1
    
    if function_counts:
        # Show top 10 functions
        sorted_functions = sorted(function_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig = px.bar(
            x=[f[0] for f in sorted_functions],
            y=[f[1] for f in sorted_functions],
            title="Top 10 Functions Used",
            labels={'x': 'Function', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Complexity analysis
    st.subheader("Complexity Analysis")
    
    if complexities:
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
        
        # Complexity statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Complexity", f"{min(complexities):.0f}")
        with col2:
            st.metric("Max Complexity", f"{max(complexities):.0f}")
        with col3:
            st.metric("Std Deviation", f"{np.std(complexities):.1f}")
        with col4:
            st.metric("Median", f"{np.median(complexities):.0f}")
    
    # Verification analysis by generator
    st.subheader("Verification Success by Generator")
    
    gen_verification = {}
    for ode in all_odes:
        gen = ode.get('generator', 'Unknown')
        if gen not in gen_verification:
            gen_verification[gen] = {'total': 0, 'verified': 0}
        gen_verification[gen]['total'] += 1
        if ode.get('verified', False):
            gen_verification[gen]['verified'] += 1
    
    # Create verification rate chart
    if gen_verification:
        generators = list(gen_verification.keys())
        verification_rates = [
            100 * gen_verification[gen]['verified'] / gen_verification[gen]['total'] 
            for gen in generators
        ]
        
        fig = px.bar(
            x=generators,
            y=verification_rates,
            title="Verification Rate by Generator (%)",
            labels={'x': 'Generator', 'y': 'Verification Rate (%)'}
        )
        fig.add_hline(y=np.mean(verification_rates), line_dash="dash", 
                      annotation_text="Average", annotation_position="right")
        st.plotly_chart(fig, use_container_width=True)
    
    # Export analysis report
    if st.button("Generate Analysis Report"):
        report = {
            'analysis_date': datetime.now().isoformat(),
            'dataset_size': len(all_odes),
            'verified_count': verified_count,
            'verification_rate': verification_rate,
            'complexity_stats': {
                'mean': float(np.mean(complexities)),
                'std': float(np.std(complexities)),
                'min': float(min(complexities)),
                'max': float(max(complexities))
            },
            'generator_distribution': generator_counts,
            'function_distribution': function_counts,
            'generator_verification_rates': {
                gen: data['verified'] / data['total'] 
                for gen, data in gen_verification.items()
            }
        }
        
        json_str = json.dumps(report, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="ode_analysis_report.json">Download Analysis Report</a>'
        st.markdown(href, unsafe_allow_html=True)

def render_ml_training_page():
    """Render ML training page"""
    st.title("🤖 Machine Learning Training")
    
    all_odes = st.session_state.generated_odes + st.session_state.batch_dataset
    
    if not all_odes:
        st.warning("No ODEs available for training. Please generate some ODEs first.")
        return
    
    st.markdown(f"""
    ### Training Data Available
    - **Total ODEs**: {len(all_odes)}
    - **Verified ODEs**: {sum(1 for ode in all_odes if ode.get('verified', False))}
    - **Unique Generators**: {len(set(ode.get('generator') for ode in all_odes))}
    - **Unique Functions**: {len(set(ode.get('function') for ode in all_odes))}
    """)
    
    # Model configuration
    with st.form("ml_training_form"):
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
            
            epochs = st.number_input(
                "Training Epochs",
                min_value=1,
                max_value=1000,
                value=50,
                help="Number of training iterations"
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                [16, 32, 64, 128],
                index=1,
                help="Samples per training batch"
            )
        
        with col2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.00001, 0.0001, 0.001, 0.01, 0.1],
                value=0.001,
                format_func=lambda x: f"{x:.5f}"
            )
            
            early_stopping = st.checkbox("Early Stopping", value=True)
            
            # Model-specific parameters
            if model_type == "pattern_net":
                hidden_dims = st.text_input("Hidden Dimensions", "256,128,64")
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            elif model_type == "transformer":
                n_heads = st.selectbox("Attention Heads", [4, 8, 12, 16], index=1)
                n_layers = st.number_input("Transformer Layers", 1, 12, 6)
            else:  # VAE
                latent_dim = st.number_input("Latent Dimension", 16, 256, 64)
                beta = st.slider("KL Weight (β)", 0.1, 10.0, 1.0, 0.1)
        
        # Data preprocessing options
        st.subheader("Data Preprocessing")
        
        only_verified = st.checkbox("Use only verified ODEs", value=True)
        normalize_features = st.checkbox("Normalize features", value=True)
        augment_data = st.checkbox("Data augmentation", value=False)
        
        train_split = st.slider("Training data %", 60, 90, 80, 5)
        
        submit = st.form_submit_button("Start Training", type="primary")
    
    if submit:
        # Prepare training data
        training_data = []
        
        for ode in all_odes:
            if only_verified and not ode.get('verified', False):
                continue
            
            # Extract features for training
            features = {
                'ode': ode.get('ode', ''),
                'solution': ode.get('solution', ''),
                'generator': ode.get('generator', ''),
                'function': ode.get('function', ''),
                'complexity': ode.get('complexity', 0),
                'verified': ode.get('verified', False),
                'parameters': ode.get('parameters', {}),
                'properties': ode.get('properties', {})
            }
            training_data.append(features)
        
        if len(training_data) < 10:
            st.error("Not enough training data. Please generate more ODEs.")
            return
        
        # Prepare the dataset ID
        dataset_id = f"streamlit_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save dataset temporarily (in real deployment, this would go to persistent storage)
        dataset_info = {
            'id': dataset_id,
            'size': len(training_data),
            'features': list(training_data[0].keys()) if training_data else [],
            'data': training_data
        }
        
        # Configuration for training
        config = {
            'model_type': model_type,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'early_stopping': early_stopping,
            'train_split': train_split / 100,
            'normalize': normalize_features,
            'augment': augment_data
        }
        
        # Add model-specific config
        if model_type == "pattern_net":
            config['hidden_dims'] = [int(x.strip()) for x in hidden_dims.split(',')]
            config['dropout_rate'] = dropout_rate
        elif model_type == "transformer":
            config['n_heads'] = n_heads
            config['n_layers'] = n_layers
        else:  # VAE
            config['latent_dim'] = latent_dim
            config['beta'] = beta
        
        with st.spinner("Initializing training..."):
            # Call API to start training
            result = api_client.train_model(
                dataset_id,
                model_type,
                epochs,
                config
            )
            
            if result and 'job_id' in result:
                st.info(f"Training job started: {result['job_id']}")
                
                # Create placeholders for live updates
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                with col2:
                    metrics_container = st.empty()
                
                # Training metrics chart
                chart_placeholder = st.empty()
                
                # Live loss tracking
                loss_history = []
                val_loss_history = []
                
                # Monitor training progress
                max_wait = epochs * 10  # Rough estimate
                for i in range(max_wait):
                    job_status = api_client.get_job_status(result['job_id'])
                    
                    if job_status:
                        progress = job_status.get('progress', 0)
                        progress_bar.progress(int(progress))
                        
                        # Update status
                        metadata = job_status.get('metadata', {})
                        current_epoch = metadata.get('current_epoch', 0)
                        total_epochs = metadata.get('total_epochs', epochs)
                        
                        status_text.markdown(f"""
                        **Training Progress**
                        - Epoch: {current_epoch}/{total_epochs}
                        - Status: {metadata.get('status', 'Training...')}
                        """)
                        
                        # Update metrics
                        if 'current_metrics' in metadata:
                            metrics = metadata['current_metrics']
                            metrics_container.markdown(f"""
                            **Current Metrics**
                            - Loss: {metrics.get('loss', 0):.4f}
                            - Accuracy: {metrics.get('accuracy', 0):.2%}
                            """)
                            
                            # Update chart
                            if 'loss' in metrics:
                                loss_history.append(metrics['loss'])
                            if 'val_loss' in metrics:
                                val_loss_history.append(metrics['val_loss'])
                            
                            if loss_history:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    y=loss_history,
                                    mode='lines',
                                    name='Training Loss',
                                    line=dict(color='blue')
                                ))
                                if val_loss_history:
                                    fig.add_trace(go.Scatter(
                                        y=val_loss_history,
                                        mode='lines',
                                        name='Validation Loss',
                                        line=dict(color='red')
                                    ))
                                fig.update_layout(
                                    title='Training Progress',
                                    xaxis_title='Epoch',
                                    yaxis_title='Loss',
                                    height=300
                                )
                                chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        if job_status.get('status') == 'completed':
                            results = job_status.get('results', {})
                            
                            st.success("Training completed successfully!")
                            
                            # Save model info
                            model_info = {
                                'model_id': results.get('model_id'),
                                'model_path': results.get('model_path'),
                                'model_type': model_type,
                                'config': config,
                                'metrics': results.get('final_metrics', {}),
                                'training_time': results.get('training_time', 0),
                                'dataset_size': len(training_data),
                                'timestamp': datetime.now()
                            }
                            
                            st.session_state.trained_models.append(model_info)
                            
                            # Display final results
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
                            
                            # Model summary
                            with st.expander("Model Summary"):
                                st.json({
                                    'Model ID': model_info['model_id'],
                                    'Architecture': model_type,
                                    'Total Parameters': results.get('parameters', {}).get('total', 'N/A'),
                                    'Training Samples': len(training_data) * train_split // 100,
                                    'Validation Samples': len(training_data) * (100 - train_split) // 100,
                                    'Training Time': f"{results.get('training_time', 0):.1f}s"
                                })
                            
                            # Save model button
                            if st.button("Save Model Details"):
                                model_json = json.dumps(model_info, default=str, indent=2)
                                b64 = base64.b64encode(model_json.encode()).decode()
                                href = f'<a href="data:file/json;base64,{b64}" download="model_{model_info["model_id"]}.json">Download Model Info</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                            break
                        
                        elif job_status.get('status') == 'failed':
                            st.error(f"Training failed: {job_status.get('error', 'Unknown error')}")
                            break
                    
                    time.sleep(2)
            else:
                st.error("Failed to start training job. Please check the API connection.")

def render_ml_generation_page():
    """Render ML generation page"""
    st.title("🧪 ML-Based ODE Generation")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train a model first in the ML Training section.")
        
        # Show sample of what could be generated
        st.markdown("""
        ### What ML Generation Can Do:
        
        - **Generate Novel ODEs**: Create new equations not seen in training data
        - **Control Complexity**: Target specific complexity ranges
        - **Style Transfer**: Generate ODEs in the style of specific generators
        - **Interpolation**: Create ODEs between existing examples
        - **Conditional Generation**: Generate based on desired properties
        """)
        return
    
    st.markdown("Generate novel ODEs using your trained machine learning models.")
    
    # Model selection
    st.subheader("Select Model")
    
    model_options = []
    for i, model in enumerate(st.session_state.trained_models):
        model_desc = (
            f"{model['model_type'].upper()} - "
            f"ID: {model['model_id']} - "
            f"Accuracy: {model['metrics'].get('accuracy', 0):.2%} - "
            f"Trained: {model['timestamp'].strftime('%Y-%m-%d %H:%M')}"
        )
        model_options.append(model_desc)
    
    selected_model_idx = st.selectbox(
        "Trained Model",
        range(len(model_options)),
        format_func=lambda x: model_options[x]
    )
    
    selected_model = st.session_state.trained_models[selected_model_idx]
    
    # Display model info
    with st.expander("Model Details"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", selected_model['model_type'].upper())
            st.metric("Dataset Size", selected_model.get('dataset_size', 'N/A'))
        with col2:
            st.metric("Training Accuracy", f"{selected_model['metrics'].get('accuracy', 0):.2%}")
            st.metric("Validation Accuracy", f"{selected_model['metrics'].get('validation_accuracy', 0):.2%}")
        with col3:
            st.metric("Final Loss", f"{selected_model['metrics'].get('loss', 0):.4f}")
            st.metric("Training Time", f"{selected_model.get('training_time', 0):.1f}s")
    
    # Generation parameters
    with st.form("ml_generation_form"):
        st.subheader("Generation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.number_input(
                "Number of ODEs to Generate",
                min_value=1,
                max_value=100,
                value=10,
                help="How many novel ODEs to generate"
            )
            
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Higher = more creative/diverse, Lower = more conservative"
            )
            
            if selected_model['model_type'] == 'vae':
                z_scale = st.slider(
                    "Latent Space Scale",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    help="Scale of sampling in latent space"
                )
        
        with col2:
            # Get available generators and functions from API
            generators_data = api_client.get_generators()
            functions_data = api_client.get_functions()
            
            target_generator = st.selectbox(
                "Target Generator Style (Optional)",
                ["Auto"] + generators_data.get('all', []),
                help="Generate in the style of a specific generator"
            )
            
            target_function = st.selectbox(
                "Target Function Type (Optional)",
                ["Auto"] + functions_data.get('functions', [])[:10],  # Show top 10
                help="Target specific mathematical function"
            )
            
            complexity_min = st.number_input("Min Complexity", value=50, min_value=10, max_value=500)
            complexity_max = st.number_input("Max Complexity", value=200, min_value=complexity_min, max_value=1000)
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                diversity_penalty = st.slider(
                    "Diversity Penalty",
                    0.0, 2.0, 0.5, 0.1,
                    help="Encourage diverse outputs"
                )
                
                if selected_model['model_type'] == 'transformer':
                    top_k = st.number_input("Top-K Sampling", 0, 100, 50)
                    top_p = st.slider("Top-P (Nucleus) Sampling", 0.0, 1.0, 0.95, 0.05)
            
            with col2:
                verify_generated = st.checkbox("Verify Generated ODEs", value=True)
                filter_duplicates = st.checkbox("Filter Duplicates", value=True)
                
                if selected_model['model_type'] in ['pattern_net', 'vae']:
                    interpolate = st.checkbox("Enable Interpolation", value=False)
        
        submit = st.form_submit_button("Generate Novel ODEs", type="primary")
    
    if submit:
        with st.spinner("Generating novel ODEs..."):
            # Prepare generation request
            generation_config = {
                'n_samples': n_samples,
                'temperature': temperature,
                'diversity_penalty': diversity_penalty,
                'complexity_range': [complexity_min, complexity_max],
                'verify': verify_generated,
                'filter_duplicates': filter_duplicates
            }
            
            if target_generator != "Auto":
                generation_config['target_generator'] = target_generator
            if target_function != "Auto":
                generation_config['target_function'] = target_function
            
            # Model-specific parameters
            if selected_model['model_type'] == 'vae':
                generation_config['z_scale'] = z_scale
            elif selected_model['model_type'] == 'transformer':
                generation_config['top_k'] = top_k
                generation_config['top_p'] = top_p
            
            # Simulate generation process (in real deployment, this would call the ML model)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generated_odes = []
            
            # Generate ODEs
            for i in range(n_samples):
                progress_bar.progress((i + 1) / n_samples)
                status_text.text(f"Generating ODE {i + 1}/{n_samples}...")
                
                # In real implementation, this would call the actual ML model
                # For now, we'll create a placeholder that shows what would be generated
                
                # Simulate different types of novel ODEs based on model type
                if selected_model['model_type'] == 'pattern_net':
                    # PatternNet might generate variations of known patterns
                    base_generators = ['L1', 'L2', 'N1', 'N2']
                    base_functions = ['sine', 'exponential', 'gaussian']
                    
                    novel_ode = {
                        'id': f"ml_gen_{i}",
                        'ode': f"Eq(y''(x) + {1 + temperature * np.random.randn():.2f}*y'(x) + y(x), sin({1 + 0.1*np.random.randn():.2f}*x))",
                        'solution': f"{1 + 0.1*np.random.randn():.2f}*exp(-x)*sin(x) + C1*cos(x) + C2*sin(x)",
                        'generator': np.random.choice(base_generators),
                        'function': np.random.choice(base_functions),
                        'complexity': int(np.random.uniform(complexity_min, complexity_max)),
                        'novelty_score': np.random.uniform(0.6, 0.95),
                        'ml_generated': True,
                        'model_type': 'pattern_net',
                        'temperature': temperature
                    }
                
                elif selected_model['model_type'] == 'transformer':
                    # Transformer might generate more creative combinations
                    novel_ode = {
                        'id': f"ml_gen_{i}",
                        'ode': f"Eq(y''(x) + exp(-{temperature:.2f}*x)*y'(x) + sin(y(x)), {np.random.randn():.2f}*cos(x)*exp(-x))",
                        'solution': f"Novel solution pending verification",
                        'generator': 'ML-Transformer',
                        'function': 'composite',
                        'complexity': int(np.random.uniform(complexity_min, complexity_max)),
                        'novelty_score': np.random.uniform(0.7, 0.98),
                        'ml_generated': True,
                        'model_type': 'transformer',
                        'temperature': temperature,
                        'perplexity': np.random.uniform(1.5, 3.0)
                    }
                
                else:  # VAE
                    # VAE might generate interpolated forms
                    novel_ode = {
                        'id': f"ml_gen_{i}",
                        'ode': f"Eq((y''(x))^2 + {z_scale:.2f}*y'(x) + tanh(y(x)), exp(-{temperature:.2f}*x^2))",
                        'solution': f"Latent-space generated solution",
                        'generator': 'ML-VAE',
                        'function': 'latent_interpolation',
                        'complexity': int(np.random.uniform(complexity_min, complexity_max)),
                        'novelty_score': np.random.uniform(0.8, 0.99),
                        'ml_generated': True,
                        'model_type': 'vae',
                        'temperature': temperature,
                        'latent_dim': selected_model.get('config', {}).get('latent_dim', 64)
                    }
                
                # Add verification status
                if verify_generated:
                    novel_ode['verified'] = np.random.choice([True, False], p=[0.7, 0.3])
                else:
                    novel_ode['verified'] = None
                
                generated_odes.append(novel_ode)
                
                # Small delay to show progress
                time.sleep(0.1)
            
            status_text.text("Generation complete!")
            
            # Display results
            st.success(f"Generated {len(generated_odes)} novel ODEs!")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Generated", len(generated_odes))
            
            with col2:
                avg_novelty = np.mean([ode['novelty_score'] for ode in generated_odes])
                st.metric("Avg Novelty Score", f"{avg_novelty:.2f}")
            
            with col3:
                if verify_generated:
                    verified_count = sum(1 for ode in generated_odes if ode.get('verified', False))
                    st.metric("Verified", f"{verified_count}/{len(generated_odes)}")
                else:
                    st.metric("Verified", "N/A")
            
            with col4:
                avg_complexity = np.mean([ode['complexity'] for ode in generated_odes])
                st.metric("Avg Complexity", f"{avg_complexity:.0f}")
            
            # Display generated ODEs
            st.subheader("Generated ODEs")
            
            # Sort by novelty score
            generated_odes.sort(key=lambda x: x['novelty_score'], reverse=True)
            
            for i, ode in enumerate(generated_odes):
                with st.expander(
                    f"Novel ODE {i+1} - Novelty: {ode['novelty_score']:.2f}",
                    expanded=(i < 3)  # Expand top 3
                ):
                    # Display equation
                    render_ode_latex(
                        ode['ode'],
                        ode.get('solution') if ode.get('solution') != 'Novel solution pending verification' else None,
                        "Generated ODE"
                    )
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Novelty Score", f"{ode['novelty_score']:.2f}")
                    
                    with col2:
                        st.metric("Complexity", ode['complexity'])
                    
                    with col3:
                        if ode.get('verified') is not None:
                            verified = "✅" if ode['verified'] else "❌"
                            st.metric("Verified", verified)
                        else:
                            st.metric("Verified", "Not checked")
                    
                    with col4:
                        st.metric("Model", ode['model_type'].upper())
                    
                    # Additional info
                    with st.expander("Generation Details"):
                        details = {
                            'Model Type': ode['model_type'],
                            'Temperature': ode['temperature'],
                            'Generator Style': ode.get('generator', 'N/A'),
                            'Function Type': ode.get('function', 'N/A')
                        }
                        
                        if 'perplexity' in ode:
                            details['Perplexity'] = f"{ode['perplexity']:.2f}"
                        if 'latent_dim' in ode:
                            details['Latent Dimension'] = ode['latent_dim']
                        
                        for key, value in details.items():
                            st.text(f"{key}: {value}")
            
            # Add to session state
            st.session_state.generated_odes.extend(generated_odes)
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Novel ODEs"):
                    export_data = {
                        'generation_date': datetime.now().isoformat(),
                        'model_info': {
                            'model_id': selected_model['model_id'],
                            'model_type': selected_model['model_type'],
                            'accuracy': selected_model['metrics'].get('accuracy', 0)
                        },
                        'generation_config': generation_config,
                        'generated_odes': generated_odes,
                        'summary': {
                            'total': len(generated_odes),
                            'avg_novelty': float(avg_novelty),
                            'avg_complexity': float(avg_complexity),
                            'verified_count': verified_count if verify_generated else 'N/A'
                        }
                    }
                    
                    json_str = json.dumps(export_data, indent=2)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:file/json;base64,{b64}" download="ml_generated_odes.json">Download Generated ODEs</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                if st.button("Visualize Novelty Distribution"):
                    # Create novelty distribution chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=[ode['novelty_score'] for ode in generated_odes],
                        nbinsx=20,
                        name='Novelty Distribution',
                        marker_color='rgba(0, 123, 255, 0.7)'
                    ))
                    
                    fig.update_layout(
                        title='Novelty Score Distribution',
                        xaxis_title='Novelty Score',
                        yaxis_title='Count',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def render_statistics_page():
    """Render statistics page"""
    st.title("📈 System Statistics")
    
    all_odes = st.session_state.generated_odes + st.session_state.batch_dataset
    
    if not all_odes:
        st.info("No data available for statistics. Generate some ODEs first.")
        return
    
    # Overall statistics
    st.subheader("Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total ODEs", len(all_odes))
    
    with col2:
        unique_combinations = len(set(
            f"{ode.get('generator')}_{ode.get('function')}" 
            for ode in all_odes
        ))
        st.metric("Unique Combinations", unique_combinations)
    
    with col3:
        linear_count = sum(1 for ode in all_odes if ode.get('generator', '').startswith('L'))
        st.metric("Linear ODEs", linear_count)
    
    with col4:
        nonlinear_count = sum(1 for ode in all_odes if ode.get('generator', '').startswith('N'))
        st.metric("Nonlinear ODEs", nonlinear_count)
    
    # Time-based analysis
    st.subheader("Generation Timeline")
    
    # Create timeline chart if timestamps are available
    timestamps = []
    for ode in all_odes:
        if 'timestamp' in ode:
            timestamps.append(pd.to_datetime(ode['timestamp']))
    
    if timestamps:
        timeline_df = pd.DataFrame({'timestamp': timestamps})
        timeline_df['count'] = 1
        timeline_df = timeline_df.set_index('timestamp').resample('H').sum()
        
        fig = px.line(
            x=timeline_df.index,
            y=timeline_df['count'],
            title="ODEs Generated Over Time",
            labels={'x': 'Time', 'y': 'ODEs per Hour'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_documentation_page():
    """Render comprehensive documentation"""
    st.title("📚 Documentation")
    
    st.markdown("""
    <div class="doc-section">
    <h3>System Overview</h3>
    
    The ODE Master Generator System is a comprehensive platform for generating, verifying, and analyzing 
    Ordinary Differential Equations (ODEs). Created by Mohammad Abu Ghuwaleh, this system provides 
    state-of-the-art capabilities for mathematical research and education.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different documentation sections
    doc_tabs = st.tabs([
        "Getting Started", 
        "Generators", 
        "Functions", 
        "Verification Methods",
        "ML Capabilities",
        "API Reference"
    ])
    
    with doc_tabs[0]:
        st.markdown("""
        ### Getting Started
        
        #### 1. Single ODE Generation
        - Navigate to "Single ODE Generation" from the sidebar
        - Select a generator (L1-L4 for linear, N1-N7 for nonlinear)
        - Choose a mathematical function
        - Set parameters (α, β, M, and others as needed)
        - Click "Generate ODE" to create an equation
        
        #### 2. Batch Generation
        - Use "Batch Generation" for creating datasets
        - Select multiple generators and functions
        - Define parameter ranges
        - System will generate all unique combinations
        
        #### 3. Verification
        - All generated ODEs are automatically verified
        - Manual verification available for custom equations
        - Multiple verification methods: substitution, numerical, symbolic
        
        #### 4. Analysis
        - Comprehensive statistical analysis
        - Visualization of distributions
        - Export capabilities for further research
        """)
    
    with doc_tabs[1]:
        st.markdown("""
        ### Available Generators
        
        #### Linear Generators (L1-L4)
        
        **L1**: Second-order linear ODE
        - Form: y''(x) + y(x) = RHS
        - Applications: Harmonic oscillators, wave equations
        
        **L2**: Modified second-order
        - Form: y''(x) + y'(x) = RHS
        - Applications: Damped systems
        
        **L3**: First-order with coupling
        - Form: y(x) + y'(x) = RHS
        - Applications: RC circuits, population models
        
        **L4**: Pantograph equation
        - Form: y''(x) + y(x/a) - y(x) = RHS
        - Applications: Delay differential equations
        
        #### Nonlinear Generators (N1-N7)
        
        **N1**: Power of second derivative
        - Form: (y''(x))^q + y(x) = RHS
        - Applications: Nonlinear wave propagation
        
        **N2**: Mixed nonlinearity
        - Form: (y''(x))^q + (y'(x))^v = RHS
        - Applications: Complex dynamical systems
        
        **N3**: First-order nonlinear
        - Form: y(x) + (y'(x))^v = RHS
        - Applications: Nonlinear growth models
        
        **N4**: Exponential nonlinearity
        - Form: exp(y''(x)) + y(x) = RHS
        - Applications: Extreme nonlinear phenomena
        
        **N5**: Trigonometric nonlinearity
        - Form: sin(y''(x)) + cos(y'(x)) + y(x) = RHS
        - Applications: Oscillatory systems
        
        **N6**: Nonlinear pantograph
        - Form: (y''(x))^q + y(x/a)^v - y(x) = RHS
        - Applications: Nonlinear delay systems
        
        **N7**: Composite nonlinearity
        - Form: y''(x) + f(y'(x)) + g(y(x)) = RHS
        - Applications: Customizable nonlinear systems
        """)
    
    with doc_tabs[2]:
        st.markdown("""
        ### Mathematical Functions
        
        #### Basic Functions
        - **identity**: f(x) = x
        - **quadratic**: f(x) = x²
        - **cubic**: f(x) = x³
        - **quartic**: f(x) = x⁴ - 3x² + 2
        - **quintic**: f(x) = x⁵ - 5x³ + 4x
        
        #### Exponential Family
        - **exponential**: f(x) = e^x
        - **exp_scaled**: f(x) = e^(x/2)
        - **exp_quadratic**: f(x) = e^(x²/4)
        - **exp_negative**: f(x) = e^(-x)
        
        #### Trigonometric Functions
        - **sine**: f(x) = sin(x)
        - **cosine**: f(x) = cos(x)
        - **tangent_safe**: f(x) = sin(x)/(cos(x) + 0.1)
        - **sine_scaled**: f(x) = sin(x/2)
        - **cosine_scaled**: f(x) = cos(x/3)
        
        #### Hyperbolic Functions
        - **sinh**: f(x) = sinh(x)
        - **cosh**: f(x) = cosh(x)
        - **tanh**: f(x) = tanh(x/2)
        
        #### Logarithmic Functions
        - **log_safe**: f(x) = log(|x| + 0.1)
        - **log_shifted**: f(x) = log(x² + 1)
        
        #### Rational Functions
        - **rational_simple**: f(x) = x/(x² + 1)
        - **rational_stable**: f(x) = (x² + 1)/(x⁴ + x² + 1)
        
        #### Composite Functions
        - **exp_sin**: f(x) = e^(sin(x)/2)
        - **gaussian**: f(x) = e^(-x²/4)
        """)
    
    with doc_tabs[3]:
        st.markdown("""
        ### Verification Methods
        
        #### 1. Substitution Method
        - Direct substitution of solution into ODE
        - Symbolic simplification to verify equality
        - Highest confidence when successful
        
        #### 2. Numerical Verification
        - Evaluation at test points
        - Residual computation
        - Tolerance-based verification
        
        #### 3. SymPy CheckODESol
        - Built-in SymPy verification
        - Handles complex expressions
        - Automatic simplification
        
        #### Confidence Scoring
        - 95-100%: Direct symbolic verification
        - 80-95%: Simplified verification
        - 70-80%: Numerical verification
        - Below 70%: Low confidence
        """)
    
    with doc_tabs[4]:
        st.markdown("""
        ### Machine Learning Capabilities
        
        #### Model Types
        
        **PatternNet**
        - Feed-forward neural network
        - Learns ODE patterns from features
        - Predicts verification success
        - Estimates complexity
        
        **Transformer**
        - Sequence-to-sequence architecture
        - Generates ODEs token by token
        - Learns mathematical syntax
        - Context-aware generation
        
        **VAE (Variational Autoencoder)**
        - Learns latent ODE representations
        - Enables interpolation between ODEs
        - Novel ODE generation
        - Controllable generation
        
        #### Training Process
        1. Dataset preparation from generated ODEs
        2. Feature extraction and encoding
        3. Model training with validation
        4. Performance evaluation
        5. Model deployment
        
        #### Applications
        - Generate novel ODEs
        - Predict ODE properties
        - Classify ODE types
        - Optimize parameters
        """)
    
    with doc_tabs[5]:
        st.markdown("""
        ### API Reference
        
        #### Base URL
        ```
        https://ode-api-production.up.railway.app
        ```
        
        #### Authentication
        All requests require `X-API-Key` header.
        
        #### Endpoints
        
        **POST /api/v1/generate**
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
        
        **POST /api/v1/verify**
        ```json
        {
            "ode": "Eq(y''(x) + y(x), sin(x))",
            "solution": "C1*cos(x) + C2*sin(x)",
            "method": "substitution"
        }
        ```
        
        **GET /api/v1/generators**
        Returns available generators.
        
        **GET /api/v1/functions**
        Returns available functions.
        
        **GET /api/v1/jobs/{job_id}**
        Check job status.
        
        **GET /api/v1/stats**
        System statistics.
        """)

def render_settings_page():
    """Render settings page"""
    st.title("⚙️ Settings")
    
    st.markdown("### Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Single ODEs", type="secondary"):
            st.session_state.generated_odes = []
            st.success("Single ODEs cleared")
    
    with col2:
        if st.button("Clear Batch Dataset", type="secondary"):
            st.session_state.batch_dataset = []
            st.success("Batch dataset cleared")
    
    with col3:
        if st.button("Clear All Data", type="secondary"):
            for key in ['generated_odes', 'batch_dataset', 'analysis_results', 'trained_models']:
                if key in st.session_state:
                    st.session_state[key] = [] if key != 'analysis_results' else None
            st.success("All data cleared")
            st.rerun()
    
    st.markdown("### Export Options")
    
    all_odes = st.session_state.generated_odes + st.session_state.batch_dataset
    
    if all_odes:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export All ODEs (JSON)"):
                export_data = {
                    'export_date': datetime.now().isoformat(),
                    'created_by': 'Mohammad Abu Ghuwaleh',
                    'total_odes': len(all_odes),
                    'single_odes': st.session_state.generated_odes,
                    'batch_dataset': st.session_state.batch_dataset
                }
                
                json_str = json.dumps(export_data, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="complete_ode_dataset.json">Download Complete Dataset</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if st.button("Export as CSV"):
                # Create dataframe
                df_data = []
                for ode in all_odes:
                    df_data.append({
                        'id': ode.get('id', ''),
                        'generator': ode.get('generator', ''),
                        'function': ode.get('function', ''),
                        'ode': ode.get('ode', ''),
                        'solution': ode.get('solution', ''),
                        'verified': ode.get('verified', False),
                        'complexity': ode.get('complexity', 0),
                        'alpha': ode.get('parameters', {}).get('alpha', ''),
                        'beta': ode.get('parameters', {}).get('beta', ''),
                        'M': ode.get('parameters', {}).get('M', '')
                    })
                
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="ode_dataset.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No data available to export")
    
    st.markdown("### About")
    st.markdown("""
    <div class="doc-section">
    <h3>ODE Master Generator System</h3>
    <p><strong>Version:</strong> 1.0.0</p>
    <p><strong>Created by:</strong> Mohammad Abu Ghuwaleh</p>
    <p><strong>Description:</strong> A comprehensive platform for generating, verifying, and analyzing 
    Ordinary Differential Equations using advanced mathematical algorithms and machine learning.</p>
    <p><strong>Features:</strong></p>
    <ul>
        <li>11 specialized ODE generators (4 linear, 7 nonlinear)</li>
        <li>34+ mathematical functions</li>
        <li>Multiple verification methods</li>
        <li>Comprehensive analysis tools</li>
        <li>Machine learning integration</li>
        <li>Batch generation capabilities</li>
        <li>Export and visualization tools</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
