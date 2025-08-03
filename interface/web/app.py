from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
import os
import json
import threading
import queue
from datetime import datetime
import pandas as pd
from pathlib import Path
import uuid

from utils.config import ConfigManager
from pipeline.generator import ODEDatasetGenerator
from utils.features import FeatureExtractor
from core.functions import AnalyticFunctionLibrary

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global storage for active generations
active_generations = {}
generation_queue = queue.Queue()

class GenerationManager:
    """Manages ODE generation sessions"""
    
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, session_id: str, config: dict):
        """Create a new generation session"""
        self.sessions[session_id] = {
            'id': session_id,
            'status': 'initializing',
            'progress': 0,
            'total': 0,
            'generated': 0,
            'verified': 0,
            'start_time': datetime.now(),
            'config': config,
            'dataset': [],
            'errors': []
        }
        
    def update_session(self, session_id: str, **kwargs):
        """Update session information"""
        if session_id in self.sessions:
            self.sessions[session_id].update(kwargs)
            
    def get_session(self, session_id: str):
        """Get session information"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        """Remove session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Initialize generation manager
gen_manager = GenerationManager()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/generate')
def generate_page():
    """Generation configuration page"""
    # Get available functions and generators
    functions = list(AnalyticFunctionLibrary.get_safe_library().keys())
    
    generators = {
        'linear': ['L1', 'L2', 'L3', 'L4'],
        'nonlinear': ['N1', 'N2', 'N3']
    }
    
    return render_template('generate.html', 
                         functions=functions,
                         generators=generators)

@app.route('/api/generate', methods=['POST'])
def start_generation():
    """Start ODE generation"""
    try:
        config = request.json
        session_id = str(uuid.uuid4())
        
        # Create session
        gen_manager.create_session(session_id, config)
        
        # Start generation in background
        thread = threading.Thread(
            target=run_generation,
            args=(session_id, config)
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Generation started'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def run_generation(session_id: str, config: dict):
    """Run ODE generation in background"""
    try:
        # Update status
        gen_manager.update_session(session_id, status='running')
        socketio.emit('status_update', {
            'session_id': session_id,
            'status': 'running'
        })
        
        # Create config manager
        config_mgr = ConfigManager()
        config_mgr.config.update(config)
        
        # Create generator with custom progress callback
        generator = ODEDatasetGenerator(config_mgr)
        
        # Override the log progress method to emit socket events
        original_log_progress = generator._log_progress
        
        def custom_log_progress(current, total):
            original_log_progress(current, total)
            
            # Update session
            gen_manager.update_session(
                session_id,
                progress=current,
                total=total,
                generated=len(generator.dataset),
                verified=sum(1 for ode in generator.dataset if ode.verified)
            )
            
            # Emit progress
            socketio.emit('progress_update', {
                'session_id': session_id,
                'progress': current,
                'total': total,
                'generated': len(generator.dataset),
                'verified': sum(1 for ode in generator.dataset if ode.verified)
            })
        
        generator._log_progress = custom_log_progress
        
        # Generate dataset
        dataset = generator.generate_dataset(
            config.get('generation', {}).get('samples_per_combo', 5)
        )
        
        # Save results
        gen_manager.update_session(
            session_id,
            status='completed',
            dataset=[ode.to_dict() for ode in dataset],
            end_time=datetime.now()
        )
        
        # Save report
        generator.save_report()
        
        # Emit completion
        socketio.emit('generation_complete', {
            'session_id': session_id,
            'total_generated': len(dataset),
            'verification_rate': 100 * sum(1 for ode in dataset if ode.verified) / len(dataset)
        })
        
    except Exception as e:
        gen_manager.update_session(
            session_id,
            status='error',
            error=str(e)
        )
        
        socketio.emit('generation_error', {
            'session_id': session_id,
            'error': str(e)
        })

@app.route('/api/status/<session_id>')
def get_status(session_id):
    """Get generation status"""
    session = gen_manager.get_session(session_id)
    
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify({
        'id': session['id'],
        'status': session['status'],
        'progress': session['progress'],
        'total': session['total'],
        'generated': session['generated'],
        'verified': session['verified'],
        'start_time': session['start_time'].isoformat()
    })

@app.route('/results/<session_id>')
def results_page(session_id):
    """Results visualization page"""
    session = gen_manager.get_session(session_id)
    
    if not session:
        return "Session not found", 404
    
    return render_template('results.html', 
                         session_id=session_id,
                         session=session)

@app.route('/api/results/<session_id>')
def get_results(session_id):
    """Get generation results"""
    session = gen_manager.get_session(session_id)
    
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    # Prepare summary statistics
    dataset = session.get('dataset', [])
    
    stats = {
        'total': len(dataset),
        'verified': sum(1 for ode in dataset if ode.get('verified')),
        'linear': sum(1 for ode in dataset if ode.get('generator_type') == 'linear'),
        'nonlinear': sum(1 for ode in dataset if ode.get('generator_type') == 'nonlinear'),
        'pantograph': sum(1 for ode in dataset if ode.get('has_pantograph')),
        'complexity_distribution': {},
        'generator_performance': {},
        'function_distribution': {}
    }
    
    # Calculate distributions
    if dataset:
        complexities = [ode.get('complexity_score', 0) for ode in dataset]
        stats['complexity_distribution'] = {
            'min': min(complexities),
            'max': max(complexities),
            'mean': sum(complexities) / len(complexities),
            'bins': pd.cut(complexities, bins=10).value_counts().to_dict()
        }
        
        # Generator performance
        from collections import Counter
        gen_counts = Counter(ode.get('generator_name') for ode in dataset)
        gen_verified = Counter(
            ode.get('generator_name') 
            for ode in dataset 
            if ode.get('verified')
        )
        
        for gen in gen_counts:
            stats['generator_performance'][gen] = {
                'total': gen_counts[gen],
                'verified': gen_verified.get(gen, 0),
                'rate': gen_verified.get(gen, 0) / gen_counts[gen] * 100
            }
        
        # Function distribution
        func_counts = Counter(ode.get('function_name') for ode in dataset)
        stats['function_distribution'] = dict(func_counts)
    
    return jsonify({
        'session': {
            'id': session['id'],
            'status': session['status'],
            'start_time': session['start_time'].isoformat(),
            'end_time': session.get('end_time', datetime.now()).isoformat()
        },
        'statistics': stats,
        'sample_odes': dataset[:10]  # First 10 ODEs as sample
    })

@app.route('/api/download/<session_id>/<format>')
def download_results(session_id, format):
    """Download results in various formats"""
    session = gen_manager.get_session(session_id)
    
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    dataset = session.get('dataset', [])
    
    if format == 'json':
        # Save as JSON
        filename = f'ode_dataset_{session_id}.json'
        filepath = Path(f'/tmp/{filename}')
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    elif format == 'csv':
        # Convert to CSV
        df = pd.DataFrame(dataset)
        filename = f'ode_dataset_{session_id}.csv'
        filepath = Path(f'/tmp/{filename}')
        
        df.to_csv(filepath, index=False)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    elif format == 'parquet':
        # Extract features and save as parquet
        from core.types import ODEInstance
        
        # Reconstruct ODEInstance objects
        ode_instances = []
        for ode_dict in dataset:
            # This is simplified - in production, properly reconstruct
            pass
        
        extractor = FeatureExtractor()
        features_df = extractor.extract_features(ode_instances)
        
        filename = f'ode_features_{session_id}.parquet'
        filepath = Path(f'/tmp/{filename}')
        
        features_df.to_parquet(filepath)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    else:
        return jsonify({'error': 'Invalid format'}), 400

@app.route('/analysis')
def analysis_page():
    """Analysis and visualization page"""
    return render_template('analysis.html')

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected to ODE Generator'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)