# scripts/monitoring_dashboard.py
"""
Real-time monitoring dashboard for ODE generation system

Benefits:
- Live metrics visualization
- Performance tracking
- Alert management
- System health monitoring
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import redis
import psutil
import json
from collections import deque
import threading
import time

# Initialize Dash app
app = dash.Dash(__name__)

# Redis connection for metrics
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Data storage for live metrics
metrics_data = {
    'timestamps': deque(maxlen=1000),
    'generation_rate': deque(maxlen=1000),
    'verification_rate': deque(maxlen=1000),
    'cpu_usage': deque(maxlen=1000),
    'memory_usage': deque(maxlen=1000),
    'error_rate': deque(maxlen=1000),
    'active_jobs': deque(maxlen=1000)
}

# Metrics collector thread
def collect_metrics():
    """Collect system metrics in background"""
    while True:
        try:
            # Current timestamp
            timestamp = datetime.now()
            
            # Get metrics from Redis
            gen_rate = float(redis_client.get('metric:generation_rate') or 0)
            ver_rate = float(redis_client.get('metric:verification_rate') or 0)
            error_rate = float(redis_client.get('metric:error_rate') or 0)
            active_jobs = int(redis_client.get('metric:active_jobs') or 0)
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Store metrics
            metrics_data['timestamps'].append(timestamp)
            metrics_data['generation_rate'].append(gen_rate)
            metrics_data['verification_rate'].append(ver_rate)
            metrics_data['cpu_usage'].append(cpu_percent)
            metrics_data['memory_usage'].append(memory_percent)
            metrics_data['error_rate'].append(error_rate)
            metrics_data['active_jobs'].append(active_jobs)
            
            time.sleep(5)  # Collect every 5 seconds
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            time.sleep(5)

# Start metrics collector
collector_thread = threading.Thread(target=collect_metrics, daemon=True)
collector_thread.start()

# Dashboard layout
app.layout = html.Div([
    html.H1('ODE Generation System Dashboard', style={'textAlign': 'center'}),
    
    # Summary cards
    html.Div([
        html.Div([
            html.H3('Total Generated'),
            html.H2(id='total-generated', children='0'),
            html.P('Last 24 hours')
        ], className='metric-card', style={'width': '23%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3('Verification Rate'),
            html.H2(id='verification-rate', children='0%'),
            html.P('Success rate')
        ], className='metric-card', style={'width': '23%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3('Active Jobs'),
            html.H2(id='active-jobs', children='0'),
            html.P('Currently processing')
        ], className='metric-card', style={'width': '23%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3('System Health'),
            html.H2(id='system-health', children='OK'),
            html.P('Overall status')
        ], className='metric-card', style={'width': '23%', 'display': 'inline-block'}),
    ], style={'margin': '20px'}),
    
    # Real-time charts
    html.Div([
        # Generation rate chart
        dcc.Graph(id='generation-rate-chart', style={'width': '50%', 'display': 'inline-block'}),
        
        # System resources chart
        dcc.Graph(id='system-resources-chart', style={'width': '50%', 'display': 'inline-block'}),
    ]),
    
    html.Div([
        # Error rate chart
        dcc.Graph(id='error-rate-chart', style={'width': '50%', 'display': 'inline-block'}),
        
        # Generator performance
        dcc.Graph(id='generator-performance-chart', style={'width': '50%', 'display': 'inline-block'}),
    ]),
    
    # Recent activities table
    html.Div([
        html.H3('Recent Activities'),
        html.Div(id='recent-activities-table')
    ], style={'margin': '20px'}),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    )
])

# Callbacks for real-time updates
@app.callback(
    [Output('total-generated', 'children'),
     Output('verification-rate', 'children'),
     Output('active-jobs', 'children'),
     Output('system-health', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_metrics(n):
    """Update summary metrics"""
    
    # Get metrics from Redis
    total_gen = redis_client.get('metric:total_generated_24h') or '0'
    ver_success = float(redis_client.get('metric:verification_success_rate') or 0)
    active = metrics_data['active_jobs'][-1] if metrics_data['active_jobs'] else 0
    
    # System health check
    cpu_usage = metrics_data['cpu_usage'][-1] if metrics_data['cpu_usage'] else 0
    memory_usage = metrics_data['memory_usage'][-1] if metrics_data['memory_usage'] else 0
    error_rate = metrics_data['error_rate'][-1] if metrics_data['error_rate'] else 0
    
    if cpu_usage > 90 or memory_usage > 90:
        health_status = 'CRITICAL'
        health_color = 'red'
    elif cpu_usage > 70 or memory_usage > 70 or error_rate > 5:
        health_status = 'WARNING'
        health_color = 'orange'
    else:
        health_status = 'HEALTHY'
        health_color = 'green'
    
    return (
        f"{int(total_gen):,}",
        f"{ver_success:.1f}%",
        str(active),
        html.Span(health_status, style={'color': health_color})
    )

@app.callback(
    Output('generation-rate-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_generation_chart(n):
    """Update generation rate chart"""
    
    if not metrics_data['timestamps']:
        return go.Figure()
    
    df = pd.DataFrame({
        'timestamp': list(metrics_data['timestamps']),
        'generation_rate': list(metrics_data['generation_rate']),
        'verification_rate': list(metrics_data['verification_rate'])
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['generation_rate'],
        mode='lines',
        name='Generation Rate',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['verification_rate'],
        mode='lines',
        name='Verification Rate',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='ODE Generation & Verification Rate',
        xaxis_title='Time',
        yaxis_title='ODEs/minute',
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output('system-resources-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_resources_chart(n):
    """Update system resources chart"""
    
    if not metrics_data['timestamps']:
        return go.Figure()
    
    df = pd.DataFrame({
        'timestamp': list(metrics_data['timestamps']),
        'cpu': list(metrics_data['cpu_usage']),
        'memory': list(metrics_data['memory_usage'])
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cpu'],
        mode='lines',
        name='CPU Usage',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['memory'],
        mode='lines',
        name='Memory Usage',
        line=dict(color='purple', width=2)
    ))
    
    # Add threshold lines
    fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                  annotation_text="Warning Threshold")
    fig.add_hline(y=90, line_dash="dash", line_color="red",
                  annotation_text="Critical Threshold")
    
    fig.update_layout(
        title='System Resource Usage',
        xaxis_title='Time',
        yaxis_title='Usage %',
        yaxis_range=[0, 100],
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output('error-rate-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_error_chart(n):
    """Update error rate chart"""
    
    if not metrics_data['timestamps']:
        return go.Figure()
    
    df = pd.DataFrame({
        'timestamp': list(metrics_data['timestamps']),
        'error_rate': list(metrics_data['error_rate'])
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['error_rate'],
        mode='lines+markers',
        name='Error Rate',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.1)'
    ))
    
    fig.update_layout(
        title='Error Rate Over Time',
        xaxis_title='Time',
        yaxis_title='Errors per minute',
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output('generator-performance-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_generator_performance(n):
    """Update generator performance chart"""
    
    # Get generator statistics from Redis
    generators = ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3', 'N7']
    success_rates = []
    avg_times = []
    
    for gen in generators:
        success = float(redis_client.get(f'metric:generator:{gen}:success_rate') or 0)
        avg_time = float(redis_client.get(f'metric:generator:{gen}:avg_time') or 0)
        success_rates.append(success)
        avg_times.append(avg_time)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=generators,
        y=success_rates,
        name='Success Rate (%)',
        marker_color='green',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=generators,
        y=avg_times,
        name='Avg Time (s)',
        marker_color='blue',
        yaxis='y2',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title='Generator Performance Comparison',
        xaxis_title='Generator',
        yaxis=dict(title='Success Rate (%)', side='left'),
        yaxis2=dict(title='Average Time (s)', side='right', overlaying='y'),
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output('recent-activities-table', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_activities_table(n):
    """Update recent activities table"""
    
    # Get recent activities from Redis
    activities = []
    for i in range(10):
        activity = redis_client.get(f'activity:recent:{i}')
        if activity:
            activities.append(json.loads(activity))
    
    if not activities:
        return html.P("No recent activities")
    
    # Create table
    rows = []
    for activity in activities:
        rows.append(html.Tr([
            html.Td(activity.get('timestamp', '')),
            html.Td(activity.get('type', '')),
            html.Td(activity.get('generator', '')),
            html.Td(activity.get('status', '')),
            html.Td(activity.get('duration', ''))
        ]))
    
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th('Timestamp'),
                html.Th('Type'),
                html.Th('Generator'),
                html.Th('Status'),
                html.Th('Duration')
            ])
        ]),
        html.Tbody(rows)
    ], style={'width': '100%'})

# CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            .metric-card {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .metric-card h2 {
                color: #2c3e50;
                margin: 10px 0;
            }
            .metric-card h3 {
                color: #7f8c8d;
                margin: 5px 0;
            }
            .metric-card p {
                color: #95a5a6;
                margin: 5px 0;
            }
            table {
                border-collapse: collapse;
                margin: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)