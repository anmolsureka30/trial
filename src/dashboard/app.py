import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import sys
import os
import numpy as np
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict
import logging
from pathlib import Path
from src.data_management.enhanced_parts_mapper import EnhancedPartsMapper
from src.fraud_detection.fraud_detector import FraudDetector

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.config.config import DB_CONFIG
from src.recommendations.smart_recommender import SmartRecommender
from src.data_management.parts_manager import PartsManager
from src.dashboard.visualization_helpers import create_parts_network, create_confidence_heatmap

class DashboardApp:
    def __init__(self):
        """Initialize dashboard components"""
        try:
            self.setup_components()
            self.load_data()
        except Exception as e:
            logger.error(f"Dashboard initialization failed: {str(e)}", exc_info=True)
            raise

    def setup_components(self):
        """Initialize dashboard components"""
        try:
            self.parts_mapper = EnhancedPartsMapper()
            self.fraud_detector = FraudDetector()
        except Exception as e:
            logger.error(f"Component setup failed: {str(e)}")
            raise

    def load_data(self):
        """Load necessary data"""
        try:
            data_file = Path('Primary_Parts_Code.csv')
            if data_file.exists():
                self.data = pd.read_csv(data_file)
                logger.info(f"Loaded {len(self.data)} records from {data_file}")
            else:
                logger.warning(f"Data file not found: {data_file}")
                self.data = pd.DataFrame()
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def show_header(self):
        """Display dashboard header"""
        st.title("🚗 Insurance Claims Dashboard")
        st.markdown("---")

    def show_metrics(self):
        """Display key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Claims", "1,234", "+12%")
        with col2:
            st.metric("Average Processing Time", "2.3 days", "-8%")
        with col3:
            st.metric("Fraud Detection Rate", "95%", "+3%")
        with col4:
            st.metric("Customer Satisfaction", "4.8/5", "+0.2")

    def show_sidebar(self):
        """Configure and display sidebar"""
        st.sidebar.title("Navigation")
        return st.sidebar.radio(
            "Select Section:",
            ["Overview", "Part Mapping", "Fraud Detection", "Analysis"]
        )

    def run(self):
        """Run the dashboard application"""
        try:
            self.show_header()
            section = self.show_sidebar()
            
            if section == "Overview":
                self.show_metrics()
                # Add overview content
            elif section == "Part Mapping":
                st.header("Part Mapping")
                # Add part mapping content
            elif section == "Fraud Detection":
                st.header("Fraud Detection")
                # Add fraud detection content
            elif section == "Analysis":
                st.header("Analysis")
                # Add analysis content
                
        except Exception as e:
            logger.error(f"Dashboard execution failed: {str(e)}", exc_info=True)
            st.error("An error occurred while running the dashboard. Please check logs for details.")

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="AutoSure Insurance Claims Dashboard",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def load_css(self):
        """Load custom CSS for styling"""
        st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --primary-color: #1f77b4;
            --secondary-color: #2ecc71;
            --background-color: #f8f9fa;
            --text-color: #2c3e50;
            --alert-color: #e74c3c;
        }

        /* Main container */
        .main {
            background-color: var(--background-color);
            color: var(--text-color);
            padding: 20px;
        }

        /* Metric cards */
        .stMetric {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 10px 0;
        }

        .stMetric label {
            color: var(--text-color);
            font-weight: 600;
        }

        .stMetric .value {
            color: var(--primary-color);
            font-size: 24px;
        }

        /* Charts container */
        .chart-container {
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 15px 0;
        }

        /* Headers */
        h1, h2, h3 {
            color: var(--text-color);
            font-weight: 600;
            margin-bottom: 20px;
        }

        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }

        /* Tables */
        .dataframe {
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
        }

        .dataframe th {
            background-color: var(--primary-color);
            color: white;
            padding: 12px;
        }

        .dataframe td {
            padding: 10px;
            border: 1px solid #ddd;
        }

        /* Alerts */
        .alert {
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }

        .alert-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .alert-warning {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }

        .alert-danger {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def initialize_components(self):
        """Initialize dashboard components"""
        try:
            self.recommender = SmartRecommender()
            self.parts_manager = PartsManager()
            
            # Load reference data
            try:
                surveyor_data = pd.read_csv('Primary_Parts_Code.csv')
                if surveyor_data.empty:
                    st.error("Error: Primary_Parts_Code.csv is empty")
                    return
                    
                self.parts_mapper.load_reference_data(surveyor_data)
                st.success("Successfully loaded parts data")
                
            except FileNotFoundError:
                st.error("Error: Primary_Parts_Code.csv not found")
                # Load sample data for demonstration
                self.load_sample_data()
                
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
        
    def load_sample_data(self):
        """Load sample data for demonstration"""
        sample_data = pd.DataFrame({
            'Surveyor Part Code': [f'P{i:03d}' for i in range(100)],
            'Surveyor Part Name': [
                'Front Bumper', 'Rear Bumper', 'Hood', 'Windshield',
                'Headlight Left', 'Headlight Right', 'Door Front Left',
                'Door Front Right', 'Fender Left', 'Fender Right'
            ] * 10
        })
        self.parts_mapper.load_reference_data(sample_data)
        st.warning("Loaded sample data for demonstration")
        
    def show_overview(self):
        """Display overview dashboard"""
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.metric_card("Active Claims", 
                           self._get_active_claims_count(),
                           self._get_claims_delta())
            
        with col2:
            self.metric_card("Fraud Alerts", 
                           self._get_fraud_alerts_count(),
                           self._get_fraud_alerts_delta(),
                           "inverse")
            
        with col3:
            self.metric_card("Processing Time", 
                           f"{self._get_avg_processing_time():.1f}h",
                           self._get_processing_time_delta())
            
        with col4:
            self.metric_card("Success Rate", 
                           f"{self._get_success_rate():.1f}%",
                           self._get_success_rate_delta())
            
        # Claims Timeline
        with st.container():
            st.subheader("📈 Claims Processing Timeline")
            self.show_claims_timeline()
            
        # Part Mapping Stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Part Mapping Accuracy")
            self.show_mapping_accuracy()
            
        with col2:
            st.subheader("💰 Cost Savings Impact")
            self.show_cost_savings()
            
    def show_part_mapping_analysis(self):
        """Display part mapping analysis with enhanced visualizations"""
        st.header("🔍 Part Mapping Analysis")
        
        # Get mapping data
        mapping_stats = self._get_mapping_statistics()
        
        # Summary metrics with improved styling
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.metric_card(
                "Mapping Accuracy",
                f"{mapping_stats['accuracy']}%",
                f"{mapping_stats['accuracy_change']}%",
                "success" if mapping_stats['accuracy'] >= 90 else "warning"
            )
        
        # Add Parts Network Graph
        st.subheader("🔗 Parts Relationships Network")
        G = create_parts_network(self.parts_mapper.data_loader.parts_relationships)
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1000, font_size=8, ax=ax)
        st.pyplot(fig)
        
        # Add Confidence Heatmap
        st.subheader("🎯 Mapping Confidence Analysis")
        if self.parts_mapper.mapping_history:
            fig = create_confidence_heatmap(self.parts_mapper.mapping_history)
            st.plotly_chart(fig, use_container_width=True)

    def show_fraud_detection(self):
        """Display fraud detection analysis"""
        st.header("🔍 Fraud Detection Analysis")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        fraud_stats = self._get_fraud_statistics()
        
        with col1:
            self.metric_card(
                "Fraud Alerts",
                str(fraud_stats['alert_count']),
                f"{fraud_stats['alert_change']}%",
                "warning"
            )
        with col2:
            self.metric_card(
                "Risk Score",
                f"{fraud_stats['risk_score']}",
                f"{fraud_stats['risk_change']}%",
                "danger" if fraud_stats['risk_score'] > 70 else "normal"
            )
        with col3:
            self.metric_card(
                "Detection Rate",
                f"{fraud_stats['detection_rate']}%",
                f"{fraud_stats['detection_change']}%",
                "success"
            )
        with col4:
            self.metric_card(
                "False Positives",
                f"{fraud_stats['false_positives']}%",
                f"{fraud_stats['false_positive_change']}%",
                "warning"
            )

        # Fraud Alerts Timeline
        st.subheader("📊 Fraud Alerts Timeline")
        alerts_data = self._get_fraud_alerts_data()
        fig = px.line(alerts_data,
                      x='date',
                      y=['high_risk', 'medium_risk', 'low_risk'],
                      title='Fraud Alerts Over Time')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Alerts",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk Distribution
        st.subheader("🎯 Risk Score Distribution")
        risk_data = self._get_risk_distribution()
        fig = px.histogram(risk_data,
                          x='risk_score',
                          nbins=20,
                          color='risk_level',
                          title='Distribution of Risk Scores')
        st.plotly_chart(fig, use_container_width=True)

    def show_impact_analysis(self):
        """Display impact analysis"""
        st.header("📊 Business Impact Analysis")
        
        # Processing Time Improvement
        st.subheader("Processing Time Improvement")
        self.show_processing_improvement()
        
        # Cost Savings Breakdown
        st.subheader("Cost Savings Breakdown")
        self.show_savings_breakdown()
        
        # ROI Metrics
        self.show_roi_metrics()
        
    # Visualization Helper Methods
    def show_claims_timeline(self):
        """Show claims timeline chart"""
        data = self._get_claims_timeline_data()
        fig = px.line(data, x='date', y='count',
                     title='Daily Claims Volume',
                     template='plotly_white')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Claims",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
    def show_mapping_accuracy(self):
        """Show mapping accuracy metrics"""
        accuracy_data = self._get_mapping_accuracy()
        fig = px.bar(accuracy_data, 
                    x='category', 
                    y='percentage',
                    color='category',
                    title='Part Mapping Accuracy')
        st.plotly_chart(fig, use_container_width=True)
        
    def show_part_names_wordcloud(self, background_color='white', colormap='viridis'):
        """Display word cloud with improved visibility"""
        try:
            text = self._get_part_names()
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color=background_color,
                colormap=colormap,
                max_words=100,
                min_font_size=10,
                max_font_size=50
            ).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            
        except Exception as e:
            st.error(f"Error generating word cloud: {str(e)}")
        
    def show_mapping_heatmap(self):
        """Show part mapping similarity heatmap"""
        similarity_matrix = self._get_similarity_matrix()
        fig = px.imshow(similarity_matrix,
                       labels=dict(x="Surveyor Parts", y="Garage Parts"),
                       color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
        
    def show_knowledge_graph(self):
        """Display part relationships graph"""
        G = self._create_knowledge_graph()
        pos = nx.spring_layout(G)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
               node_size=1500, font_size=8, ax=ax)
        st.pyplot(fig)
        
    def show_anomaly_detection(self):
        """Show anomaly detection scatter plot"""
        anomaly_data = self._get_anomaly_data()
        fig = px.scatter(anomaly_data,
                        x='cost',
                        y='frequency',
                        color='is_anomaly',
                        title='Cost vs. Frequency Anomalies')
        st.plotly_chart(fig, use_container_width=True)
        
    def metric_card(self, label: str, value: str, delta: str, status: str = "normal", delta_color: str = None):
        """Display a custom metric card with improved styling"""
        colors = {
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545",
            "normal": "#1f77b4"
        }
        
        color = colors.get(status, colors["normal"])
        delta_style = f"color: {delta_color}" if delta_color else f"color: {color}"
        
        st.markdown(f"""
            <div class="stMetric" style="border-left: 5px solid {color}">
                <label>{label}</label>
                <div class="value" style="color: {color}">{value}</div>
                <div class="delta" style="{delta_style}">{delta}</div>
            </div>
        """, unsafe_allow_html=True)

    # Add other visualization methods...
    
    # Data retrieval methods (implement these based on your data structure)
    def _get_mapping_accuracy(self) -> pd.DataFrame:
        """Get mapping accuracy data"""
        # Implement actual data retrieval
        return pd.DataFrame({
            'category': ['Exact Match', 'Fuzzy Match', 'Manual Review'],
            'percentage': [75, 15, 10]
        })
        
    def _get_part_names(self) -> str:
        """Get concatenated part names for word cloud"""
        # Implement actual data retrieval
        return " ".join(['Bumper', 'Hood', 'Windshield'] * 10)
        
    def _get_similarity_matrix(self) -> np.ndarray:
        """Get part name similarity matrix"""
        # Implement actual data retrieval
        return np.random.rand(10, 10)
        
    def _create_knowledge_graph(self) -> nx.Graph:
        """Create knowledge graph of part relationships"""
        G = nx.Graph()
        # Add nodes and edges based on your data
        G.add_edge("Bumper", "Front Bumper")
        G.add_edge("Bumper", "Rear Bumper")
        return G
        
    def _get_anomaly_data(self) -> pd.DataFrame:
        """Get anomaly detection data"""
        # Implement actual data retrieval
        return pd.DataFrame({
            'cost': np.random.uniform(100, 1000, 100),
            'frequency': np.random.randint(1, 10, 100),
            'is_anomaly': np.random.choice(['Normal', 'Anomaly'], 100)
        })

    # Helper methods for data retrieval
    def _get_active_claims_count(self) -> int:
        """Get count of active claims"""
        # Implement actual database query
        return 150
        
    def _get_claims_delta(self) -> int:
        """Get change in claims count"""
        return 5
        
    def _get_fraud_alerts_count(self) -> int:
        """Get count of fraud alerts"""
        return 12
        
    def _get_fraud_alerts_delta(self) -> int:
        """Get change in fraud alerts"""
        return -2
        
    def _get_avg_processing_time(self) -> float:
        """Get average claim processing time"""
        return 4.5
        
    def _get_processing_time_delta(self) -> float:
        """Get change in processing time"""
        return -0.5
        
    def _get_success_rate(self) -> float:
        """Get claims processing success rate"""
        return 95.5
        
    def _get_success_rate_delta(self) -> float:
        """Get change in success rate"""
        return 1.2
        
    def _get_claims_timeline_data(self) -> pd.DataFrame:
        """Get claims timeline data"""
        try:
            # Sample data - replace with actual database query
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now())
            data = pd.DataFrame({
                'date': dates,
                'count': np.random.randint(10, 50, size=len(dates))
            })
            return data
        except Exception as e:
            st.error(f"Error loading claims timeline data: {str(e)}")
            return pd.DataFrame()

    def _get_recent_alerts(self) -> pd.DataFrame:
        """Get recent fraud alerts"""
        try:
            return self.fraud_detector.get_historical_alerts(days=7)
        except Exception as e:
            st.error(f"Error loading recent alerts: {str(e)}")
            return pd.DataFrame()

    def _get_claims_data(self, start_date, end_date) -> pd.DataFrame:
        """Get claims data for analysis"""
        try:
            # Sample data - replace with actual database query
            dates = pd.date_range(start=start_date, end=end_date)
            data = pd.DataFrame({
                'date': dates,
                'status': np.random.choice(['pending', 'approved', 'rejected'], size=len(dates)),
                'claim_amount': np.random.uniform(1000, 10000, size=len(dates)),
                'part_code': np.random.choice(['1001', '1002', '1003'], size=len(dates))
            })
            return data
        except Exception as e:
            st.error(f"Error loading claims data: {str(e)}")
            return pd.DataFrame()

    def _get_parts_distribution(self, claims_data: pd.DataFrame) -> pd.DataFrame:
        """Get parts distribution data"""
        try:
            parts_count = claims_data['part_code'].value_counts().reset_index()
            parts_count.columns = ['part_code', 'count']
            
            # Get part names from parts manager
            parts_count['part_name'] = parts_count['part_code'].apply(
                lambda x: self.parts_manager.get_part_by_code(x)['part_name'] 
                if self.parts_manager.get_part_by_code(x) else f'Part {x}'
            )
            
            return parts_count
        except Exception as e:
            st.error(f"Error analyzing parts distribution: {str(e)}")
            return pd.DataFrame()

    def _get_risk_distribution(self) -> pd.DataFrame:
        """Get risk score distribution data"""
        scores = np.concatenate([
            np.random.normal(30, 10, 500),  # Low risk
            np.random.normal(60, 10, 300),  # Medium risk
            np.random.normal(85, 5, 200)    # High risk
        ])
        scores = np.clip(scores, 0, 100)
        
        return pd.DataFrame({
            'risk_score': scores,
            'risk_level': ['High' if s >= 75 else 'Medium' if s >= 50 else 'Low' for s in scores]
        })

    def _get_high_risk_claims(self) -> pd.DataFrame:
        """Get high-risk claims data"""
        try:
            # Sample data - replace with actual database query
            data = pd.DataFrame({
                'claim_id': [f'CLM{i:03d}' for i in range(5)],
                'risk_score': np.random.uniform(80, 100, size=5),
                'alert_type': ['suspicious_pattern', 'high_amount', 'frequency', 'duration', 'location'],
                'status': 'pending'
            })
            return data
        except Exception as e:
            st.error(f"Error loading high-risk claims: {str(e)}")
            return pd.DataFrame()

    def _get_parts_frequency(self) -> pd.DataFrame:
        """Get parts replacement frequency data"""
        try:
            # Sample data - replace with actual database query
            parts_data = pd.DataFrame({
                'part_code': ['1001', '1002', '1003', '1004', '1005'],
                'part_name': ['Bumper', 'Hood', 'Windshield', 'Door', 'Fender'],
                'frequency': np.random.randint(10, 100, size=5)
            })
            return parts_data
        except Exception as e:
            st.error(f"Error loading parts frequency: {str(e)}")
            return pd.DataFrame()

    def _get_parts_cost_analysis(self) -> pd.DataFrame:
        """Get parts cost analysis data"""
        try:
            # Sample data - replace with actual database query
            parts_data = pd.DataFrame({
                'part_code': ['1001', '1002', '1003', '1004', '1005'],
                'part_name': ['Bumper', 'Hood', 'Windshield', 'Door', 'Fender'],
                'frequency': np.random.randint(10, 100, size=5),
                'avg_cost': np.random.uniform(500, 5000, size=5)
            })
            return parts_data
        except Exception as e:
            st.error(f"Error loading parts cost analysis: {str(e)}")
            return pd.DataFrame()

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        return 45.5  # Replace with actual CPU monitoring

    def _get_cpu_usage_delta(self) -> float:
        """Get CPU usage change"""
        return -2.3  # Replace with actual monitoring

    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        return 62.8  # Replace with actual memory monitoring

    def _get_memory_usage_delta(self) -> float:
        """Get memory usage change"""
        return 1.5  # Replace with actual monitoring

    def _get_avg_response_time(self) -> float:
        """Get average response time"""
        return 0.45  # Replace with actual response time monitoring

    def _get_response_time_delta(self) -> float:
        """Get response time change"""
        return -0.05  # Replace with actual monitoring

    def _get_performance_timeline(self) -> pd.DataFrame:
        """Get system performance timeline data"""
        try:
            dates = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
            data = pd.DataFrame({
                'timestamp': dates,
                'cpu_usage': np.random.uniform(30, 70, size=len(dates)),
                'memory_usage': np.random.uniform(50, 80, size=len(dates))
            })
            return data
        except Exception as e:
            st.error(f"Error loading performance timeline: {str(e)}")
            return pd.DataFrame()

    def show_cost_savings(self):
        """Display cost savings visualization"""
        savings_data = self._get_cost_savings_data()
        
        # Create a waterfall chart for cost savings
        fig = go.Figure(go.Waterfall(
            name="Cost Savings",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Fraud Prevention", "Efficient Processing", "Accurate Mapping", "Total Savings"],
            textposition="outside",
            text=["$" + str(x) for x in savings_data['values']],
            y=savings_data['values'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#ef553b"}},
            increasing={"marker": {"color": "#00b4d8"}},
            totals={"marker": {"color": "#2ecc71"}}
        ))
        
        fig.update_layout(
            title="Cost Savings Breakdown",
            showlegend=False,
            waterfallgap=0.2
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_cost_variance(self):
        """Display cost variance analysis"""
        variance_data = self._get_cost_variance_data()
        
        # Create box plot for cost distribution
        fig = px.box(variance_data, 
                     x="part_category", 
                     y="cost",
                     color="part_category",
                     title="Cost Distribution by Part Category")
        
        fig.update_layout(
            xaxis_title="Part Category",
            yaxis_title="Cost (USD)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add trend analysis
        col1, col2 = st.columns(2)
        with col1:
            self.show_cost_trends()
        with col2:
            self.show_outlier_analysis()

    def show_processing_improvement(self):
        """Display processing time improvements"""
        improvement_data = self._get_processing_improvement_data()
        
        # Create before/after comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=improvement_data['category'],
            y=improvement_data['before'],
            name='Before AI',
            marker_color='#e74c3c'
        ))
        
        fig.add_trace(go.Bar(
            x=improvement_data['category'],
            y=improvement_data['after'],
            name='After AI',
            marker_color='#2ecc71'
        ))
        
        fig.update_layout(
            title="Processing Time Improvement",
            barmode='group',
            xaxis_title="Process Category",
            yaxis_title="Time (hours)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def show_savings_breakdown(self):
        """Display detailed savings breakdown"""
        savings_data = self._get_detailed_savings_data()
        
        # Create pie chart for savings distribution
        fig = px.pie(savings_data, 
                     values='amount', 
                     names='category',
                     title='Cost Savings Distribution',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    def show_roi_metrics(self):
        """Display ROI metrics"""
        roi_data = self._get_roi_data()
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "ROI",
                f"{roi_data['roi']}%",
                f"{roi_data['roi_change']}%"
            )
        
        with col2:
            st.metric(
                "Cost Reduction",
                f"{roi_data['cost_reduction']}%",
                f"{roi_data['cost_reduction_change']}%"
            )
        
        with col3:
            st.metric(
                "Efficiency Gain",
                f"{roi_data['efficiency']}%",
                f"{roi_data['efficiency_change']}%"
            )

    def show_cost_trends(self):
        """Display cost trends over time"""
        trend_data = self._get_cost_trend_data()
        
        fig = px.line(trend_data,
                      x='date',
                      y='average_cost',
                      title='Average Cost Trends')
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Cost (USD)"
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_outlier_analysis(self):
        """Display cost outlier analysis"""
        outlier_data = self._get_outlier_data()
        
        fig = px.scatter(outlier_data,
                         x='claim_id',
                         y='cost',
                         color='is_outlier',
                         title='Cost Outlier Analysis')
        
        fig.update_layout(
            xaxis_title="Claim ID",
            yaxis_title="Cost (USD)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Add these data retrieval methods

    def _get_cost_savings_data(self) -> Dict:
        """Get cost savings data"""
        return {
            'values': [50000, 30000, 20000, 100000]
        }

    def _get_cost_variance_data(self) -> pd.DataFrame:
        """Get cost variance data"""
        categories = ['Body', 'Engine', 'Electronics', 'Interior']
        means = [1000, 2000, 1500, 800]
        scales = [200, 400, 300, 150]
        
        # Generate costs for each category
        all_costs = np.concatenate([
            np.random.normal(loc=mu, scale=scale, size=25)
            for mu, scale in zip(means, scales)
        ])
        
        # Create category labels
        all_categories = np.repeat(categories, 25)
        
        return pd.DataFrame({
            'part_category': all_categories,
            'cost': all_costs
        })

    def _get_processing_improvement_data(self) -> pd.DataFrame:
        """Get processing improvement data"""
        return pd.DataFrame({
            'category': ['Verification', 'Assessment', 'Approval', 'Payment'],
            'before': [24, 48, 12, 36],
            'after': [2, 4, 1, 3]
        })

    def _get_detailed_savings_data(self) -> pd.DataFrame:
        """Get detailed savings breakdown"""
        return pd.DataFrame({
            'category': ['Fraud Prevention', 'Process Optimization', 'Accurate Mapping', 'Time Savings'],
            'amount': [300000, 200000, 150000, 350000]
        })

    def _get_roi_data(self) -> Dict:
        """Get ROI metrics"""
        return {
            'roi': 156,
            'roi_change': 23,
            'cost_reduction': 35,
            'cost_reduction_change': 5,
            'efficiency': 68,
            'efficiency_change': 12
        }

    def _get_cost_trend_data(self) -> pd.DataFrame:
        """Get cost trend data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
        return pd.DataFrame({
            'date': dates,
            'average_cost': np.random.normal(1500, 200, size=len(dates))
        })

    def _get_outlier_data(self) -> pd.DataFrame:
        """Get outlier analysis data"""
        n_points = 100
        costs = np.concatenate([
            np.random.normal(1500, 200, size=90),
            np.random.normal(5000, 500, size=10)
        ])
        return pd.DataFrame({
            'claim_id': [f'CLM{i:03d}' for i in range(n_points)],
            'cost': costs,
            'is_outlier': ['Normal' if c < 3000 else 'Outlier' for c in costs]
        })

    def _get_mapping_statistics(self) -> Dict:
        """Get current mapping statistics"""
        return {
            'accuracy': 95.5,
            'accuracy_change': 2.3,
            'daily_count': 1250,
            'daily_change': 45,
            'avg_confidence': 88.7,
            'confidence_change': 1.5,
            'manual_reviews': 125,
            'review_change': -15
        }

    def _get_part_frequency_data(self) -> pd.DataFrame:
        """Get frequency data for most common parts"""
        return pd.DataFrame({
            'part_name': [
                'Front Bumper', 'Windshield', 'Headlight', 'Hood',
                'Side Mirror', 'Rear Bumper', 'Door Panel',
                'Fender', 'Tail Light', 'Grille'
            ],
            'frequency': sorted([
                np.random.randint(100, 1000) for _ in range(10)
            ], reverse=True)
        })

    def _get_confidence_distribution(self) -> pd.DataFrame:
        """Get distribution of mapping confidence scores"""
        # Generate realistic confidence scores with a skew towards higher values
        scores = np.concatenate([
            np.random.normal(85, 10, 700),  # High confidence
            np.random.normal(60, 15, 200),  # Medium confidence
            np.random.normal(30, 10, 100)   # Low confidence
        ])
        scores = np.clip(scores, 0, 100)  # Ensure values are between 0 and 100
        return pd.DataFrame({'confidence': scores})

    def _get_mapping_trend_data(self) -> pd.DataFrame:
        """Get mapping success rate trend data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        # Generate realistic trend data with gradual improvement
        base_success = np.linspace(85, 95, len(dates)) + np.random.normal(0, 1, len(dates))
        manual_review = np.linspace(15, 5, len(dates)) + np.random.normal(0, 1, len(dates))
        
        return pd.DataFrame({
            'date': dates,
            'automatic_success': base_success,
            'manual_review': manual_review
        })

    def _get_mapping_challenges(self) -> pd.DataFrame:
        """Get distribution of mapping challenges"""
        return pd.DataFrame({
            'challenge_type': [
                'Regional Variations',
                'Technical vs Common Names',
                'Misspellings',
                'Incomplete Descriptions',
                'Multiple Matches'
            ],
            'count': [300, 250, 200, 150, 100]
        })

    def show_regional_variations(self):
        """Display analysis of regional part name variations"""
        # Create sample data for regional variations
        variations_data = pd.DataFrame({
            'standard_name': ['Windshield', 'Hood', 'Trunk'],
            'us_name': ['Windshield', 'Hood', 'Trunk'],
            'uk_name': ['Windscreen', 'Bonnet', 'Boot'],
            'au_name': ['Windscreen', 'Bonnet', 'Boot'],
            'match_rate': [98, 95, 92]
        })
        
        # Display as a formatted table
        st.write("Regional Name Variations and Match Rates")
        st.dataframe(
            variations_data.style.background_gradient(subset=['match_rate'], cmap='RdYlGn'),
            hide_index=True
        )

    def _get_fraud_statistics(self) -> Dict:
        """Get fraud detection statistics"""
        return {
            'alert_count': 125,
            'alert_change': 15,
            'risk_score': 65,
            'risk_change': -5,
            'detection_rate': 92,
            'detection_change': 3,
            'false_positives': 8,
            'false_positive_change': -2
        }

    def _get_fraud_alerts_data(self) -> pd.DataFrame:
        """Get fraud alerts timeline data"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        return pd.DataFrame({
            'date': dates,
            'high_risk': np.random.randint(5, 15, size=len(dates)),
            'medium_risk': np.random.randint(10, 30, size=len(dates)),
            'low_risk': np.random.randint(20, 50, size=len(dates))
        })

if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.run() 