import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

def create_parts_network(parts_relationships):
    """Create network graph of parts relationships"""
    G = nx.Graph()
    
    # Add nodes and edges
    for location, parts in parts_relationships.items():
        G.add_node(location, node_type='location')
        for part in parts:
            G.add_node(part, node_type='part')
            G.add_edge(location, part)
    
    return G

def create_confidence_heatmap(mapping_history):
    """Create heatmap of mapping confidence scores"""
    df = pd.DataFrame(mapping_history)
    confidence_matrix = df.pivot_table(
        values='mapped_result.confidence_score',
        index='input_description',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=confidence_matrix.values,
        x=confidence_matrix.columns,
        y=confidence_matrix.index,
        colorscale='RdYlGn'
    ))
    
    return fig 

def create_claims_overview(claims_data: pd.DataFrame) -> go.Figure:
    """Create claims overview visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Claims by Status",
            "Claims by Vehicle Type",
            "Damage Severity Distribution",
            "Processing Time vs Amount"
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Claims by Status
    status_counts = claims_data['status'].value_counts()
    fig.add_trace(
        go.Pie(labels=status_counts.index, values=status_counts.values),
        row=1, col=1
    )
    
    # Claims by Vehicle Type
    vehicle_counts = claims_data['vehicle_type'].value_counts()
    fig.add_trace(
        go.Bar(x=vehicle_counts.index, y=vehicle_counts.values),
        row=1, col=2
    )
    
    # Damage Severity
    severity_counts = claims_data['damage_severity'].value_counts()
    fig.add_trace(
        go.Bar(x=severity_counts.index, y=severity_counts.values),
        row=2, col=1
    )
    
    # Processing Time vs Amount
    fig.add_trace(
        go.Scatter(
            x=claims_data['processing_time'],
            y=claims_data['total_amount'],
            mode='markers',
            marker=dict(
                color=claims_data['fraud_risk'],
                colorscale='Viridis',
                showscale=True
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Claims Overview Dashboard"
    )
    return fig

def create_fraud_analysis(claims_data: pd.DataFrame) -> go.Figure:
    """Create fraud analysis visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Fraud Risk Distribution",
            "Risk vs Amount",
            "Average Risk by Vehicle Type",
            "Risk Over Time"
        )
    )
    
    # Fraud Risk Distribution
    fig.add_trace(
        go.Histogram(x=claims_data['fraud_risk'], nbinsx=30),
        row=1, col=1
    )
    
    # Risk vs Amount
    fig.add_trace(
        go.Scatter(
            x=claims_data['total_amount'],
            y=claims_data['fraud_risk'],
            mode='markers',
            marker=dict(
                color=claims_data['processing_time'],
                colorscale='Viridis',
                showscale=True
            )
        ),
        row=1, col=2
    )
    
    # Average Risk by Vehicle Type
    avg_risk = claims_data.groupby('vehicle_type')['fraud_risk'].mean()
    fig.add_trace(
        go.Bar(x=avg_risk.index, y=avg_risk.values),
        row=2, col=1
    )
    
    # Risk Over Time
    monthly_risk = claims_data.groupby(pd.Grouper(key='timestamp', freq='M'))['fraud_risk'].mean()
    fig.add_trace(
        go.Scatter(x=monthly_risk.index, y=monthly_risk.values, mode='lines+markers'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig 