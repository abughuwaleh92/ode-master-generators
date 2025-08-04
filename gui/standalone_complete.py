import streamlit as st
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
from typing import List, Dict, Optional
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="ODE Master Generator - Complete System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import ODE modules with error handling
try:
    from pipeline.generator import ODEDatasetGenerator
    from verification.verifier import ODEVerifier
    
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Error importing modules: {e}")
    st.info("Make sure all project files are present")

# Initialize session state
if 'generated_odes' not in st.session_state:
    st.session_state.generated_odes = []
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = []
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

class StandaloneODEInterface:
    """Complete ODE System without external API"""
    
    def __init__(self):
        self.generator = ODEGenerator() if MODULES_AVAILABLE else None
        self.verifier = ODEVerifier() if MODULES_AVAILABLE else None
        
    def run(self):
        """Main application"""
        
        # Sidebar
        with st.sidebar:
            st.title("🔬 ODE Master Generator")
            st.markdown("### Navigation")
            
            page = st.selectbox(
                "Select Page",
                ["🧮 Generate ODEs", 
                 "🤖 ML Training", 
                 "🧠 AI Generation",
                 "📊 Analysis Dashboard",
                 "📁 Dataset Manager",
                 "⚡ Batch Processing",
                 "🔧 Settings"]
            )
            
            st.markdown("---")
            st.markdown("### System Status")
            st.success("✅ All Systems Operational")
            st.info(f"Generated ODEs: {len(st.session_state.generated_odes)}")
            st.info(f"Dataset Size: {len(st.session_state.current_dataset)}")
            
            st.markdown("---")
            if st.button("🗑️ Clear Session", type="secondary"):
                st.session_state.generated_odes = []
                st.session_state.current_dataset = []
                st.experimental_rerun()
        
        # Route to pages
        if page == "🧮 Generate ODEs":
            self.generation_page()
        elif page == "🤖 ML Training":
            self.ml_training_page()
        elif page == "🧠 AI Generation":
            self.ai_generation_page()
        elif page == "📊 Analysis Dashboard":
            self.analysis_dashboard()
        elif page == "📁 Dataset Manager":
            self.dataset_manager()
        elif page == "⚡ Batch Processing":
            self.batch_processing_page()
        elif page == "🔧 Settings":
            self.settings_page()
    
    def generation_page(self):
        """ODE Generation Interface"""
        st.title("🧮 ODE Generation")
        
        if not MODULES_AVAILABLE:
            st.error("ODE modules not available")
            return
        
        # Generation settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            generator = st.selectbox(
                "Generator Type",
                ["L1", "L2", "L3", "L4", "N1", "N2", "N3", "N7"],
                help="L: Linear, N: Nonlinear"
            )
            
        with col2:
            function = st.selectbox(
                "Function Type",
                ["identity", "quadratic", "sine", "cosine", "exponential", 
                 "logarithm", "rational", "bessel"],
                help="Base function for ODE"
            )
            
        with col3:
            count = st.number_input(
                "Number of ODEs",
                min_value=1,
                max_value=100,
                value=5,
                help="How many ODEs to generate"
            )
        
        # Parameters
        st.subheader("Parameters")
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            alpha = st.slider("α (Alpha)", -5.0, 5.0, 1.0, 0.1)
        with param_col2:
            beta = st.slider("β (Beta)", 0.1, 5.0, 1.0, 0.1)
        with param_col3:
            M = st.slider("M", -5.0, 5.0, 0.0, 0.1)
        
        # Advanced options
        with st.expander("Advanced Options"):
            verify = st.checkbox("Verify generated ODEs", value=True)
            show_solutions = st.checkbox("Show solutions", value=True)
            complexity_filter = st.slider(
                "Complexity Range",
                0, 500,
                (0, 500),
                help="Filter by complexity score"
            )
        
        # Generate button
        if st.button("🚀 Generate ODEs", type="primary"):
            with st.spinner(f"Generating {count} ODEs..."):
                progress_bar = st.progress(0)
                generated = []
                
                for i in range(count):
                    try:
                        # Generate ODE
                        params = {"alpha": alpha, "beta": beta, "M": M}
                        ode_data = self.generator.generate_single(generator, function, params)
                        
                        if ode_data:
                            # Verify if requested
                            if verify and ode_data.get('solution_symbolic'):
                                verification = self.verifier.verify(
                                    ode_data['ode_symbolic'],
                                    ode_data['solution_symbolic']
                                )
                                ode_data['verified'] = verification.get('verified', False)
                            
                            generated.append(ode_data)
                        
                        progress_bar.progress((i + 1) / count)
                        
                    except Exception as e:
                        st.error(f"Error generating ODE {i+1}: {str(e)}")
                
                # Store results
                st.session_state.generated_odes.extend(generated)
                
                st.success(f"✅ Generated {len(generated)} ODEs successfully!")
                
                # Display results
                self._display_generated_odes(generated, show_solutions)
    
    def _display_generated_odes(self, odes: List[Dict], show_solutions: bool = True):
        """Display generated ODEs"""
        st.subheader("Generated ODEs")
        
        for i, ode in enumerate(odes):
            with st.expander(f"ODE {i+1} - {ode.get('generator_name', 'Unknown')}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Display ODE
                    st.markdown("**Differential Equation:**")
                    st.latex(ode.get('ode_latex', ode.get('ode_symbolic', 'N/A')))
                    
                    if show_solutions and ode.get('solution_symbolic'):
                        st.markdown("**Solution:**")
                        st.latex(ode.get('solution_latex', ode.get('solution_symbolic', 'N/A')))
                
                with col2:
                    # Metrics
                    st.metric("Complexity", ode.get('complexity_score', 'N/A'))
                    st.metric("Order", ode.get('order', 'N/A'))
                    
                    if ode.get('verified') is not None:
                        if ode['verified']:
                            st.success("✓ Verified")
                        else:
                            st.error("✗ Not Verified")
                    
                    # Actions
                    if st.button(f"Add to Dataset", key=f"add_{i}"):
                        st.session_state.current_dataset.append(ode)
                        st.success("Added to dataset!")
    
    def ml_training_page(self):
        """ML Training Interface"""
        st.title("🤖 Machine Learning Training")
        
        # Check if we have data
        if len(st.session_state.current_dataset) == 0:
            st.warning("No data in dataset. Generate some ODEs first!")
            return
        
        st.info(f"Training on {len(st.session_state.current_dataset)} ODEs")
        
        # Model configuration
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["Pattern Recognition Network", "Transformer", "VAE", "Simple RNN"],
                help="Select ML model architecture"
            )
            
            epochs = st.number_input("Epochs", 1, 1000, 50)
            batch_size = st.number_input("Batch Size", 8, 256, 32)
            
        with col2:
            learning_rate = st.number_input(
                "Learning Rate",
                0.00001, 0.1, 0.001,
                format="%.5f"
            )
            
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
            early_stopping = st.checkbox("Early Stopping", value=True)
        
        # Training button
        if st.button("🎯 Start Training", type="primary"):
            # Simulate training
            st.info("Training simulation (would train real model with ML pipeline)")
            
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()
            
            # Simulate training progress
            losses = []
            for epoch in range(epochs):
                # Simulate loss
                loss = 1.0 * np.exp(-epoch/20) + np.random.normal(0, 0.05)
                losses.append(loss)
                
                # Update progress
                progress_bar.progress((epoch + 1) / epochs)
                
                # Update chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(losses))),
                    y=losses,
                    mode='lines',
                    name='Training Loss'
                ))
                fig.update_layout(
                    title="Training Progress",
                    xaxis_title="Epoch",
                    yaxis_title="Loss"
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                time.sleep(0.1)  # Simulate training time
            
            st.success("✅ Training completed!")
            
            # Save to history
            st.session_state.training_history.append({
                "model": model_type,
                "epochs": epochs,
                "final_loss": losses[-1],
                "timestamp": datetime.now()
            })
    
    def ai_generation_page(self):
        """AI-Powered Generation"""
        st.title("🧠 AI-Powered ODE Generation")
        
        st.info("This simulates AI generation based on learned patterns")
        
        generation_mode = st.radio(
            "Generation Mode",
            ["Random Exploration", "Guided Generation", "Style Transfer"]
        )
        
        if generation_mode == "Guided Generation":
            col1, col2 = st.columns(2)
            with col1:
                target_type = st.selectbox("Target Type", ["Linear", "Nonlinear", "Mixed"])
            with col2:
                complexity_target = st.slider("Target Complexity", 0, 500, 100)
        
        n_samples = st.number_input("Number to Generate", 1, 50, 10)
        
        if st.button("🎨 Generate with AI", type="primary"):
            with st.spinner("AI is creating novel ODEs..."):
                # Simulate AI generation
                ai_odes = []
                
                for i in range(n_samples):
                    # Create synthetic "AI-generated" ODE
                    ai_ode = {
                        "ode_symbolic": f"y''(x) + {np.random.rand():.2f}*y'(x) + {np.random.rand():.2f}*y(x) = AI_func(x)",
                        "generator_name": "AI",
                        "complexity_score": np.random.randint(50, 200),
                        "novelty_score": np.random.rand(),
                        "ai_generated": True
                    }
                    ai_odes.append(ai_ode)
                
                # Display results
                st.success(f"Generated {len(ai_odes)} novel ODEs!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_novelty = np.mean([ode['novelty_score'] for ode in ai_odes])
                    st.metric("Avg Novelty", f"{avg_novelty:.2%}")
                with col2:
                    avg_complexity = np.mean([ode['complexity_score'] for ode in ai_odes])
                    st.metric("Avg Complexity", f"{avg_complexity:.0f}")
                with col3:
                    st.metric("Success Rate", "100%")
                
                # Show ODEs
                for i, ode in enumerate(ai_odes[:5]):  # Show first 5
                    st.write(f"**AI ODE {i+1}:** {ode['ode_symbolic']}")
    
    def analysis_dashboard(self):
        """Analysis Dashboard"""
        st.title("📊 Analysis Dashboard")
        
        if len(st.session_state.current_dataset) == 0:
            st.warning("No data to analyze. Generate some ODEs first!")
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
            if 'complexity_score' in df.columns:
                st.metric("Avg Complexity", f"{df['complexity_score'].mean():.1f}")
        with col4:
            unique_generators = df['generator_name'].nunique() if 'generator_name' in df.columns else 0
            st.metric("Generators Used", unique_generators)
        
        # Visualizations
        if 'generator_name' in df.columns:
            st.subheader("Generator Distribution")
            fig1 = px.bar(
                df['generator_name'].value_counts().reset_index(),
                x='index',
                y='generator_name',
                title="ODEs by Generator"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        if 'complexity_score' in df.columns:
            st.subheader("Complexity Distribution")
            fig2 = px.histogram(
                df,
                x='complexity_score',
                nbins=30,
                title="Complexity Score Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed table
        st.subheader("Dataset Overview")
        st.dataframe(
            df[['generator_name', 'function_name', 'complexity_score', 'verified']]
            if all(col in df.columns for col in ['generator_name', 'function_name', 'complexity_score', 'verified'])
            else df
        )
    
    def dataset_manager(self):
        """Dataset Management"""
        st.title("📁 Dataset Manager")
        
        tab1, tab2, tab3 = st.tabs(["Current Dataset", "Import/Export", "Merge Datasets"])
        
        with tab1:
            st.subheader("Current Dataset")
            st.info(f"Current dataset contains {len(st.session_state.current_dataset)} ODEs")
            
            if st.session_state.current_dataset:
                # Actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Clear Dataset", type="secondary"):
                        st.session_state.current_dataset = []
                        st.success("Dataset cleared!")
                
                # Show first few entries
                st.subheader("Preview (First 5 entries)")
                for i, ode in enumerate(st.session_state.current_dataset[:5]):
                    st.write(f"{i+1}. {ode.get('ode_symbolic', 'N/A')}")
        
        with tab2:
            st.subheader("Export Dataset")
            
            if st.session_state.current_dataset:
                # Export format
                export_format = st.selectbox("Export Format", ["JSON Lines", "JSON", "CSV"])
                
                if st.button("📥 Export Dataset"):
                    if export_format == "JSON Lines":
                        # Create JSONL
                        jsonl_str = '\n'.join([json.dumps(ode) for ode in st.session_state.current_dataset])
                        st.download_button(
                            label="Download JSONL",
                            data=jsonl_str,
                            file_name=f"ode_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                            mime="application/jsonl"
                        )
                    
                    elif export_format == "JSON":
                        json_str = json.dumps(st.session_state.current_dataset, indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"ode_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    elif export_format == "CSV":
                        df = pd.DataFrame(st.session_state.current_dataset)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"ode_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            st.subheader("Import Dataset")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['jsonl', 'json', 'csv']
            )
            
            if uploaded_file is not None:
                if st.button("📤 Import Dataset"):
                    try:
                        if uploaded_file.name.endswith('.jsonl'):
                            lines = uploaded_file.read().decode('utf-8').strip().split('\n')
                            data = [json.loads(line) for line in lines]
                        elif uploaded_file.name.endswith('.json'):
                            data = json.load(uploaded_file)
                        elif uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                            data = df.to_dict('records')
                        
                        st.session_state.current_dataset.extend(data)
                        st.success(f"Imported {len(data)} ODEs!")
                        
                    except Exception as e:
                        st.error(f"Error importing: {str(e)}")
    
    def batch_processing_page(self):
        """Batch Processing"""
        st.title("⚡ Batch Processing")
        
        st.subheader("Batch Generation")
        
        # Batch configuration
        col1, col2 = st.columns(2)
        
        with col1:
            generators = st.multiselect(
                "Generators",
                ["L1", "L2", "L3", "L4", "N1", "N2", "N3", "N7"],
                default=["L1", "L2"]
            )
            
        with col2:
            functions = st.multiselect(
                "Functions",
                ["sine", "cosine", "exponential", "quadratic"],
                default=["sine", "cosine"]
            )
        
        samples_per_combo = st.number_input(
            "Samples per combination",
            1, 100, 10,
            help=f"Will generate {len(generators) * len(functions) * 10} ODEs total"
        )
        
        total_odes = len(generators) * len(functions) * samples_per_combo
        st.info(f"Total ODEs to generate: {total_odes}")
        
        if st.button("⚡ Start Batch Generation", type="primary"):
            if not MODULES_AVAILABLE:
                st.error("ODE modules not available")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generated_count = 0
            total_combinations = len(generators) * len(functions)
            
            for i, gen in enumerate(generators):
                for j, func in enumerate(functions):
                    status_text.text(f"Generating {gen} with {func}...")
                    
                    for k in range(samples_per_combo):
                        try:
                            ode_data = self.generator.generate_single(gen, func)
                            if ode_data:
                                st.session_state.current_dataset.append(ode_data)
                                generated_count += 1
                        except:
                            pass
                        
                        progress = generated_count / total_odes
                        progress_bar.progress(progress)
            
            st.success(f"✅ Batch generation complete! Generated {generated_count} ODEs")
    
    def settings_page(self):
        """Settings Page"""
        st.title("🔧 Settings")
        
        st.subheader("Display Settings")
        
        show_latex = st.checkbox("Show LaTeX rendering", value=True)
        show_symbolic = st.checkbox("Show symbolic expressions", value=True)
        
        st.subheader("Generation Settings")
        
        default_verify = st.checkbox("Verify by default", value=True)
        default_count = st.number_input("Default generation count", 1, 100, 5)
        
        st.subheader("Performance Settings")
        
        parallel_processing = st.checkbox("Enable parallel processing", value=False)
        cache_results = st.checkbox("Cache results", value=True)
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved! (In a real app, these would persist)")

# Run the application
if __name__ == "__main__":
    if MODULES_AVAILABLE:
        app = StandaloneODEInterface()
        app.run()
    else:
        st.error("Cannot load ODE modules. Please check your installation.")
        st.info("""
        To fix this:
        1. Make sure all project files are present
        2. Check that Python path is correct
        3. Install required dependencies
        """)
