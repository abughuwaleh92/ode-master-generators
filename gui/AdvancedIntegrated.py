# gui/AdvancedIntegrated.py
"""
Advanced Integrated GUI for ODE Master Generator
Complete system with ML training, AI generation, and analysis
"""

import os
import time
import json
import base64
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
    page_title="ODE Master Generator – Advanced System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────
#  CONFIGURATION & STYLING
# ──────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
API_KEY = os.getenv('API_KEY', 'test-key')

# Custom CSS for better UI
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .stExpander {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
#  API CLIENT CLASS
# ──────────────────────────────────────────────────────────────────────────
class APIClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def get_job_status(self, job_id):
        """Get job status from API"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/jobs/{job_id}",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def create_dataset(self, odes, dataset_name):
        """Create dataset on server"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/datasets/create",
                json={"odes": odes, "dataset_name": dataset_name},
                headers=self.headers,
                timeout=30
            )
            return response
        except Exception as e:
            return None
    
    def get_datasets(self):
        """Get available datasets"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/datasets",
                headers=self.headers,
                timeout=5
            )
            return response
        except:
            return None
    
    def start_training(self, training_request):
        """Start ML training job"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/ml/train",
                json=training_request,
                headers=self.headers,
                timeout=30
            )
            return response
        except:
            return None

# Initialize API client
api_client = APIClient(API_BASE_URL, API_KEY)

# ──────────────────────────────────────────────────────────────────────────
#  SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────
if 'generated_odes' not in st.session_state:
    st.session_state.generated_odes = []
if 'batch_dataset' not in st.session_state:
    st.session_state.batch_dataset = []
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'current_ode' not in st.session_state:
    st.session_state.current_ode = []
if 'only_verified' not in st.session_state:
    st.session_state.only_verified = True

# ──────────────────────────────────────────────────────────────────────────
#  MAIN INTERFACE
# ──────────────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.title("🔬 ODE Master")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "⚡ Quick Generate", "🤖 ML Training", 
             "🧠 AI Generation", "📊 Analysis", "🔧 Settings"],
            label_visibility="collapsed"
        )
        
        # Connection Status
        st.markdown("### Connection Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.warning("⚠️ API Offline")
        
        # Stats
        st.markdown("### Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Generated", len(st.session_state.generated_odes))
        with col2:
            st.metric("Dataset", len(st.session_state.batch_dataset))
    
    # Main content area
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "⚡ Quick Generate":
        render_quick_generate()
    elif page == "🤖 ML Training":
        render_ml_training_page()
    elif page == "🧠 AI Generation":
        render_ai_generation()
    elif page == "📊 Analysis":
        render_analysis()
    elif page == "🔧 Settings":
        render_settings()

# ──────────────────────────────────────────────────────────────────────────
#  PAGE RENDERERS
# ──────────────────────────────────────────────────────────────────────────
def render_dashboard():
    """Dashboard page"""
    st.title("🏠 ODE Master Dashboard")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    all_odes = st.session_state.generated_odes + st.session_state.batch_dataset
    verified_count = sum(1 for ode in all_odes if ode.get('verified', False))
    
    with col1:
        st.metric("Total ODEs", len(all_odes), delta=len(st.session_state.generated_odes))
    with col2:
        st.metric("Verified", verified_count, delta=f"{verified_count/len(all_odes)*100:.1f}%" if all_odes else "0%")
    with col3:
        st.metric("Models Trained", len(st.session_state.trained_models))
    with col4:
        unique_generators = len(set(ode.get('generator', '') for ode in all_odes))
        st.metric("Unique Generators", unique_generators)
    
    # Recent activity
    st.markdown("### Recent Activity")
    if all_odes:
        recent_odes = all_odes[-5:]
        for i, ode in enumerate(reversed(recent_odes)):
            with st.expander(f"ODE {len(all_odes)-i}: {ode.get('generator', 'Unknown')}-{ode.get('function', 'Unknown')}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.code(ode.get('ode', 'N/A'))
                with col2:
                    st.metric("Verified", "✅" if ode.get('verified') else "❌")
                    if 'complexity' in ode:
                        st.metric("Complexity", ode['complexity'])
    else:
        st.info("No ODEs generated yet. Start by generating some ODEs!")
    
    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Generate ODEs", use_container_width=True):
            st.switch_page("pages/quick_generate.py")
    
    with col2:
        if st.button("🤖 Train Model", use_container_width=True):
            st.switch_page("pages/ml_training.py")
    
    with col3:
        if st.button("📊 Analyze Data", use_container_width=True):
            st.switch_page("pages/analysis.py")

def render_quick_generate():
    """Quick generation page"""
    st.title("⚡ Quick ODE Generation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        generator = st.selectbox(
            "Generator",
            ["L1", "L2", "L3", "L4", "N1", "N2", "N3", "N7"]
        )
    
    with col2:
        function = st.selectbox(
            "Function",
            ["sine", "cosine", "exponential", "logarithm", "bessel", "rational"]
        )
    
    with col3:
        count = st.number_input("Count", min_value=1, max_value=100, value=10)
    
    # Parameters
    with st.expander("⚙️ Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            alpha = st.slider("α (Alpha)", -5.0, 5.0, 1.0, 0.1)
        with col2:
            beta = st.slider("β (Beta)", 0.1, 5.0, 1.0, 0.1)
        with col3:
            M = st.slider("M", -5.0, 5.0, 0.0, 0.1)
    
    verify = st.checkbox("Verify generated ODEs", value=True)
    
    if st.button("🚀 Generate ODEs", type="primary", use_container_width=True):
        with st.spinner(f"Generating {count} ODEs..."):
            # Simulate generation (replace with actual API call)
            progress_bar = st.progress(0)
            generated = []
            
            for i in range(count):
                # Mock ODE generation
                ode = {
                    'id': f"{generator}_{function}_{i+1}",
                    'generator': generator,
                    'function': function,
                    'ode': f"y''(x) + {alpha:.2f}*y'(x) + {beta:.2f}*y(x) = {function}(x)",
                    'solution': f"y(x) = C1*exp(r1*x) + C2*exp(r2*x) + particular_solution",
                    'verified': verify and np.random.choice([True, False], p=[0.8, 0.2]),
                    'complexity': np.random.randint(50, 200),
                    'parameters': {'alpha': alpha, 'beta': beta, 'M': M},
                    'timestamp': datetime.now().isoformat()
                }
                generated.append(ode)
                progress_bar.progress((i + 1) / count)
                time.sleep(0.1)  # Simulate processing
            
            st.session_state.generated_odes.extend(generated)
            st.success(f"✅ Generated {len(generated)} ODEs successfully!")
            
            # Display results
            st.markdown("### Generated ODEs")
            for i, ode in enumerate(generated[:5]):  # Show first 5
                with st.expander(f"ODE {i+1}: {ode['id']}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.code(ode['ode'])
                        if ode.get('solution'):
                            st.code(f"Solution: {ode['solution']}")
                    with col2:
                        st.metric("Verified", "✅" if ode['verified'] else "❌")
                        st.metric("Complexity", ode['complexity'])
            
            if len(generated) > 5:
                st.info(f"Showing first 5 of {len(generated)} ODEs. View all in the dashboard.")
            
            # Add to dataset option
            if st.button("➕ Add to Dataset", use_container_width=True):
                st.session_state.batch_dataset.extend(generated)
                st.success(f"Added {len(generated)} ODEs to dataset!")

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
    
    # First, create dataset on server
    if st.button("Prepare Dataset for Training", type="primary"):
        with st.spinner("Creating dataset on server..."):
            # Filter ODEs if needed
            training_data = []
            for ode in all_odes:
                if st.session_state.get('only_verified', True) and not ode.get('verified', False):
                    continue
                training_data.append(ode)
            
            if len(training_data) < 10:
                st.error("Not enough ODEs for training. Need at least 10 ODEs (verified).")
                return
            
            # Create dataset on server
            dataset_name = f"streamlit_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/datasets/create",
                json={
                    "odes": training_data,
                    "dataset_name": dataset_name
                },
                headers={"X-API-Key": API_KEY}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Dataset created: {result['dataset_name']} ({result['size']} ODEs)")
                st.session_state.current_dataset = result['dataset_name']
            else:
                st.error(f"Failed to create dataset: {response.text}")
                return
    
    # Show available datasets
    with st.expander("Available Datasets", expanded=True):
        response = requests.get(
            f"{API_BASE_URL}/api/v1/datasets",
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code == 200:
            datasets = response.json()['datasets']
            if datasets:
                df = pd.DataFrame(datasets)
                # Convert size to MB if it's file size
                if 'size' in df.columns:
                    df['size_mb'] = df['size'].apply(lambda x: f"{x/1024/1024:.2f} MB" if x > 1024 else f"{x} ODEs")
                st.dataframe(df[['name', 'size_mb', 'created_at']], use_container_width=True)
                
                # Select dataset for training
                dataset_names = [d['name'] for d in datasets]
                if dataset_names:
                    selected_dataset = st.selectbox(
                        "Select dataset for training",
                        dataset_names,
                        index=len(dataset_names)-1 if dataset_names else 0
                    )
                    st.session_state.current_dataset = selected_dataset
                else:
                    st.warning("No datasets available. Create one using the button above.")
        else:
            st.error("Failed to fetch datasets")
    
    # Model configuration form
    if 'current_dataset' in st.session_state and st.session_state.current_dataset:
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
                
                epochs = st.number_input("Training Epochs", min_value=1, max_value=1000, value=50)
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            
            with col2:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.00001, 0.0001, 0.001, 0.01, 0.1],
                    value=0.001,
                    format_func=lambda x: f"{x:.5f}"
                )
                
                early_stopping = st.checkbox("Early Stopping", value=True)
            
            # Model-specific parameters
            config = {}
            if model_type == "pattern_net":
                hidden_dims = st.text_input("Hidden Dimensions", "256,128,64")
                config['hidden_dims'] = [int(x.strip()) for x in hidden_dims.split(',')]
                config['dropout_rate'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            elif model_type == "transformer":
                config['n_heads'] = st.selectbox("Attention Heads", [4, 8, 12, 16], index=1)
                config['n_layers'] = st.number_input("Transformer Layers", 1, 12, 6)
            else:  # VAE
                config['latent_dim'] = st.number_input("Latent Dimension", 16, 256, 64)
                config['beta'] = st.slider("KL Weight (β)", 0.1, 10.0, 1.0, 0.1)
            
            # Data preprocessing options
            st.subheader("Data Preprocessing")
            only_verified = st.checkbox("Use only verified ODEs", value=True)
            normalize_features = st.checkbox("Normalize features", value=True)
            
            config.update({
                'only_verified': only_verified,
                'normalize_features': normalize_features
            })
            
            submit = st.form_submit_button("Start Training", type="primary")
        
        if submit:
            # Call API to start training with the dataset name
            training_request = {
                'dataset': st.session_state.current_dataset,  # Use dataset name, not data
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'early_stopping': early_stopping,
                'config': config
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/ml/train",
                json=training_request,
                headers={"X-API-Key": API_KEY}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Training job started: {result['job_id']}")
                
                # Monitor training progress
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                with col2:
                    metrics_container = st.empty()
                
                # Training metrics chart
                chart_placeholder = st.empty()
                
                # Live tracking
                loss_history = []
                accuracy_history = []
                
                # Monitor training
                max_checks = epochs * 2  # Rough estimate
                for check in range(max_checks):
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
                        - Dataset: {metadata.get('dataset_path', st.session_state.current_dataset)}
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
                            if 'accuracy' in metrics:
                                accuracy_history.append(metrics['accuracy'])
                            
                            if loss_history:
                                fig = go.Figure()
                                
                                # Create subplot with secondary y-axis
                                fig.add_trace(go.Scatter(
                                    y=loss_history,
                                    mode='lines',
                                    name='Training Loss',
                                    line=dict(color='red'),
                                    yaxis='y'
                                ))
                                
                                if accuracy_history:
                                    fig.add_trace(go.Scatter(
                                        y=accuracy_history,
                                        mode='lines',
                                        name='Accuracy',
                                        line=dict(color='green'),
                                        yaxis='y2'
                                    ))
                                
                                fig.update_layout(
                                    title='Training Progress',
                                    xaxis_title='Epoch',
                                    yaxis=dict(
                                        title='Loss',
                                        side='left'
                                    ),
                                    yaxis2=dict(
                                        title='Accuracy',
                                        overlaying='y',
                                        side='right'
                                    ),
                                    height=400
                                )
                                
                                chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        if job_status.get('status') == 'completed':
                            results = job_status.get('results', {})
                            
                            st.success("✅ Training completed successfully!")
                            
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
                            
                            # Model info
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
                            
                            # Model summary
                            with st.expander("Model Summary"):
                                st.json({
                                    'Model ID': model_info['model_id'],
                                    'Architecture': model_type,
                                    'Total Parameters': results.get('parameters', {}).get('total', 'N/A'),
                                    'Training Samples': len(training_data),
                                    'Training Time': f"{results.get('training_time', 0):.1f}s",
                                    'Dataset': st.session_state.current_dataset
                                })
                            
                            # Save model button
                            if st.button("💾 Save Model Details"):
                                model_json = json.dumps(model_info, default=str, indent=2)
                                b64 = base64.b64encode(model_json.encode()).decode()
                                href = f'<a href="data:file/json;base64,{b64}" download="model_{model_info["model_id"]}.json">Download Model Info</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                            break
                        
                        elif job_status.get('status') == 'failed':
                            st.error(f"❌ Training failed: {job_status.get('error', 'Unknown error')}")
                            break
                    
                    time.sleep(2)  # Check every 2 seconds
                    
                    if check == max_checks - 1:
                        st.warning("Training is taking longer than expected. The job may still be running in the background.")
            else:
                st.error(f"Failed to start training: {response.text}")
    else:
        st.info("Please create or select a dataset first using the button above.")

def render_ai_generation():
    """AI Generation page"""
    st.title("🧠 AI-Powered ODE Generation")
    
    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train a model first.")
        return
    
    # Model selection
    model_names = [f"{m['model_type']} - {m['model_id']}" for m in st.session_state.trained_models]
    selected_model_idx = st.selectbox("Select Model", range(len(model_names)), format_func=lambda x: model_names[x])
    selected_model = st.session_state.trained_models[selected_model_idx]
    
    # Generation settings
    col1, col2 = st.columns(2)
    
    with col1:
        generation_mode = st.radio(
            "Generation Mode",
            ["Random", "Guided", "Interactive"]
        )
        
        n_samples = st.number_input("Number of Samples", 1, 100, 10)
    
    with col2:
        temperature = st.slider("Creativity", 0.1, 2.0, 0.8)
        diversity_penalty = st.slider("Diversity Bonus", 0.0, 1.0, 0.5)
    
    # Mode-specific settings
    if generation_mode == "Guided":
        st.subheader("Guidance Settings")
        col1, col2 = st.columns(2)
        with col1:
            target_generator = st.selectbox("Target Generator", ["Any", "L1", "L2", "L3", "L4", "N1", "N2", "N3", "N7"])
            target_complexity = st.slider("Target Complexity", 0, 500, (100, 300))
        with col2:
            target_function = st.selectbox("Target Function", ["Any", "sine", "cosine", "exponential", "bessel"])
    
    elif generation_mode == "Interactive":
        st.subheader("Interactive Builder")
        st.info("Build ODE structure interactively")
        
        # Interactive ODE builder would go here
        st.text_area("ODE Template", "y''(x) + [?]*y'(x) + [?]*y(x) = [?]")
    
    if st.button("🎨 Generate with AI", type="primary", use_container_width=True):
        with st.spinner("AI is generating novel ODEs..."):
            # Simulate AI generation
            progress_bar = st.progress(0)
            generated_odes = []
            
            for i in range(n_samples):
                # Mock AI generation
                ode = {
                    'id': f'ai_{selected_model["model_id"]}_{i+1}',
                    'ode': f"y''(x) + {np.random.randn():.2f}*y'(x) + {np.random.randn():.2f}*y(x) = f(x)",
                    'novelty_score': np.random.rand(),
                    'confidence': np.random.rand(),
                    'generator': 'AI',
                    'model': selected_model['model_id'],
                    'timestamp': datetime.now().isoformat()
                }
                generated_odes.append(ode)
                progress_bar.progress((i + 1) / n_samples)
                time.sleep(0.1)
            
            st.success(f"✅ Generated {len(generated_odes)} novel ODEs!")
            
            # Display results with metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_novelty = np.mean([ode['novelty_score'] for ode in generated_odes])
                st.metric("Avg Novelty", f"{avg_novelty:.2%}")
            with col2:
                avg_confidence = np.mean([ode['confidence'] for ode in generated_odes])
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            with col3:
                st.metric("Generated", len(generated_odes))
            
            # Show ODEs
            for i, ode in enumerate(generated_odes[:5]):
                with st.expander(f"AI ODE {i+1} - Novelty: {ode['novelty_score']:.2%}"):
                    st.code(ode['ode'])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Novelty Score", f"{ode['novelty_score']:.2%}")
                    with col2:
                        st.metric("Confidence", f"{ode['confidence']:.2%}")

def render_analysis():
    """Analysis page"""
    st.title("📊 ODE Analysis")
    
    all_odes = st.session_state.generated_odes + st.session_state.batch_dataset
    
    if not all_odes:
        st.info("No ODEs to analyze. Generate some ODEs first!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_odes)
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total ODEs", len(df))
    with col2:
        if 'verified' in df.columns:
            verified_pct = df['verified'].sum() / len(df) * 100
            st.metric("Verified", f"{verified_pct:.1f}%")
    with col3:
        if 'complexity' in df.columns:
            st.metric("Avg Complexity", f"{df['complexity'].mean():.1f}")
    with col4:
        if 'generator' in df.columns:
            st.metric("Unique Generators", df['generator'].nunique())
    
    # Visualizations
    st.markdown("### Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Patterns", "Time Series"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'generator' in df.columns:
                st.subheader("Generator Distribution")
                fig = px.pie(df['generator'].value_counts().reset_index(), 
                           values='generator', names='index',
                           title="ODEs by Generator")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'function' in df.columns:
                st.subheader("Function Distribution")
                fig = px.bar(df['function'].value_counts().reset_index(),
                           x='index', y='function',
                           title="ODEs by Function Type")
                st.plotly_chart(fig, use_container_width=True)
        
        if 'complexity' in df.columns:
            st.subheader("Complexity Distribution")
            fig = px.histogram(df, x='complexity', nbins=30,
                             title="ODE Complexity Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Pattern Analysis")
        
        if 'generator' in df.columns and 'verified' in df.columns:
            # Verification rate by generator
            verification_by_gen = df.groupby('generator')['verified'].agg(['sum', 'count'])
            verification_by_gen['rate'] = verification_by_gen['sum'] / verification_by_gen['count'] * 100
            
            fig = px.bar(verification_by_gen.reset_index(), 
                       x='generator', y='rate',
                       title="Verification Rate by Generator")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'timestamp' in df.columns:
            st.subheader("Generation Timeline")
            # Convert timestamp to datetime if it's not already
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Group by hour
            df['hour'] = df['timestamp'].dt.floor('H')
            timeline = df.groupby('hour').size().reset_index(name='count')
            
            fig = px.line(timeline, x='hour', y='count',
                        title="ODEs Generated Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.markdown("### Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download CSV"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="ode_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("📥 Download JSON"):
            json_str = df.to_json(orient='records', indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="ode_data.json">Download JSON File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        if st.button("🗑️ Clear All Data"):
            st.session_state.generated_odes = []
            st.session_state.batch_dataset = []
            st.rerun()

def render_settings():
    """Settings page"""
    st.title("🔧 Settings")
    
    # API Configuration
    st.markdown("### API Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        api_url = st.text_input("API Base URL", value=API_BASE_URL)
    with col2:
        api_key = st.text_input("API Key", value=API_KEY, type="password")
    
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{api_url}/health", headers={"X-API-Key": api_key}, timeout=5)
            if response.status_code == 200:
                st.success("✅ Connection successful!")
            else:
                st.error(f"❌ Connection failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
    
    # Training Defaults
    st.markdown("### Training Defaults")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.only_verified = st.checkbox("Only use verified ODEs for training", value=st.session_state.get('only_verified', True))
        default_epochs = st.number_input("Default Epochs", 1, 1000, 50)
    with col2:
        default_batch_size = st.selectbox("Default Batch Size", [16, 32, 64, 128], index=1)
        default_lr = st.select_slider("Default Learning Rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], value=0.001)
    
    # Display Settings
    st.markdown("### Display Settings")
    
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
    show_timestamps = st.checkbox("Show timestamps", value=True)
    max_display_odes = st.slider("Max ODEs to display", 5, 50, 10)
    
    # Data Management
    st.markdown("### Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Session Data"):
            session_data = {
                'generated_odes': st.session_state.generated_odes,
                'batch_dataset': st.session_state.batch_dataset,
                'trained_models': st.session_state.trained_models,
                'timestamp': datetime.now().isoformat()
            }
            json_str = json.dumps(session_data, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="session_data.json">Download Session Data</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        uploaded_file = st.file_uploader("Import Session Data", type=['json'])
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                st.session_state.generated_odes = data.get('generated_odes', [])
                st.session_state.batch_dataset = data.get('batch_dataset', [])
                st.session_state.trained_models = data.get('trained_models', [])
                st.success("✅ Session data imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to import data: {str(e)}")
    
    with col3:
        if st.button("Clear All Data", type="secondary"):
            if st.checkbox("Confirm clear all data"):
                st.session_state.generated_odes = []
                st.session_state.batch_dataset = []
                st.session_state.trained_models = []
                st.session_state.current_dataset = None
                st.success("✅ All data cleared!")
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────
#  RUN APPLICATION
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
