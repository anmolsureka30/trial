import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import requests
import json
from datetime import datetime, timedelta
import logging
from src.dashboard.analytics_module import ClaimsAnalytics
from src.analytics.advanced_predictions import AdvancedPredictions
from src.reports.report_generator import ReportGenerator
from typing import List, Dict
from src.dashboard.visualization_helpers import create_claims_overview, create_fraud_analysis
import itertools
import numpy as np
from scipy import stats
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EnhancedDashboard:
    def __init__(self):
        """Initialize the enhanced dashboard"""
        self.api_base_url = "http://localhost:8000/api"
        self.setup_page_config()
        self.load_data()

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Insurance Claims Dashboard",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def _initialize_statistics(self) -> Dict:
        """Initialize default statistics"""
        return {
            "total_claims": 0,
            "approved_claims": 0,
            "rejected_claims": 0,
            "under_review": 0,
            "average_fraud_risk": 0,
            "total_amount": 0,
            "new_claims": 0,
            "processing_time": 0
        }

    def load_data(self):
        """Load necessary data for dashboard"""
        try:
            # Initialize default statistics
            self.statistics = self._initialize_statistics()
            
            # Load local data first
            claims_path = Path("data/historical_claims.csv")
            if claims_path.exists():
                self.claims_data = pd.read_csv(claims_path)
                # Convert timestamp to datetime
                self.claims_data['timestamp'] = pd.to_datetime(self.claims_data['timestamp'])
                
                # Calculate statistics from data
                today = pd.Timestamp.now().date()
                self.statistics.update({
                    "total_claims": len(self.claims_data),
                    "approved_claims": len(self.claims_data[self.claims_data['status'] == 'approved']),
                    "rejected_claims": len(self.claims_data[self.claims_data['status'] == 'rejected']),
                    "under_review": len(self.claims_data[self.claims_data['status'] == 'pending']),
                    "average_fraud_risk": self.claims_data['fraud_risk'].mean(),
                    "total_amount": self.claims_data['total_amount'].sum(),
                    "new_claims": len(self.claims_data[self.claims_data['timestamp'].dt.date == today]),
                    "processing_time": self.claims_data['processing_time'].mean()
                })
            else:
                self.claims_data = pd.DataFrame()
            
            # Try to update with API data
            try:
                response = requests.get(f"{self.api_base_url}/claims/statistics", timeout=2)
                if response.status_code == 200:
                    self.statistics.update(response.json())
            except requests.exceptions.RequestException as e:
                logger.warning(f"API connection failed: {e}")
                st.warning("API connection failed. Using local data only.")
            
        except Exception as e:
            logger.error(f"Failed to load dashboard data: {e}")
            st.error("Failed to load dashboard data. Please check the logs.")

    def run(self):
        """Run the dashboard"""
        try:
            self.show_header()
            
            # Sidebar navigation
            page = st.sidebar.radio(
                "Navigation",
                ["Overview", "Parts Analysis", "Mapping Analysis", "Fraud Analytics", "Trend Analysis"]
            )
            
            if page == "Overview":
                self.show_overview()
            elif page == "Parts Analysis":
                self.show_parts_analysis()
            elif page == "Mapping Analysis":
                self.show_mapping_analysis()
            elif page == "Fraud Analytics":
                self.show_fraud_analytics()
            else:
                self.show_trend_analysis()
                
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            st.error("An error occurred. Please check the logs.")

    def show_header(self):
        """Display dashboard header"""
        st.title("🚗 Insurance Claims Dashboard")
        st.markdown("---")

    def show_overview(self):
        """Display overview page"""
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Claims",
                f"{self.statistics['total_claims']:,}",
                f"+{self.statistics['new_claims']} today"
            )
        with col2:
            approval_rate = (self.statistics['approved_claims'] / self.statistics['total_claims'] * 100) 
            st.metric(
                "Approval Rate",
                f"{approval_rate:.1f}%",
                f"{approval_rate - 75:.1f}% vs target"
            )
        with col3:
            st.metric(
                "Average Fraud Risk",
                f"{self.statistics['average_fraud_risk']:.1f}%",
                "-2.1% vs last month"
            )
        with col4:
            st.metric(
                "Total Amount",
                f"${self.statistics['total_amount']:,.2f}",
                "+12.3% vs last month"
            )

        # Claims Overview
        st.subheader("Claims Overview")
        overview_fig = create_claims_overview(self.claims_data)
        st.plotly_chart(overview_fig, use_container_width=True)
        
        # Claims Timeline
        st.subheader("Claims Timeline")
        timeline_data = self.claims_data.groupby(
            pd.Grouper(key='timestamp', freq='D')
        ).agg({
            'claim_id': 'count',
            'total_amount': 'sum',
            'fraud_risk': 'mean'
        }).reset_index()
        
        timeline_fig = go.Figure()
        timeline_fig.add_trace(go.Scatter(
            x=timeline_data['timestamp'],
            y=timeline_data['claim_id'],
            name='Number of Claims',
            line=dict(color='blue')
        ))
        timeline_fig.add_trace(go.Scatter(
            x=timeline_data['timestamp'],
            y=timeline_data['total_amount'] / 1000,  # Convert to thousands
            name='Total Amount (K)',
            yaxis='y2',
            line=dict(color='green')
        ))
        
        timeline_fig.update_layout(
            title='Claims and Amounts Over Time',
            yaxis=dict(title='Number of Claims'),
            yaxis2=dict(title='Total Amount (K)', overlaying='y', side='right')
        )
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Fraud Analysis
        st.subheader("Fraud Analysis")
        fraud_fig = create_fraud_analysis(self.claims_data)
        st.plotly_chart(fraud_fig, use_container_width=True)
        
        # Vehicle Type Distribution
        st.subheader("Claims by Vehicle Type")
        vehicle_data = self.claims_data['vehicle_type'].value_counts()
        vehicle_fig = px.pie(
            values=vehicle_data.values,
            names=vehicle_data.index,
            title='Distribution of Claims by Vehicle Type'
        )
        st.plotly_chart(vehicle_fig, use_container_width=True)

    def show_parts_analysis(self):
        """Display parts analysis page"""
        st.subheader("Parts Analysis Dashboard")
        
        # Most Commonly Damaged Parts
        st.markdown("### Most Commonly Damaged Parts")
        parts_data = pd.DataFrame([
            item for sublist in self.claims_data['parts_affected'].apply(eval) 
            for item in sublist
        ], columns=['part_name'])
        
        parts_count = parts_data['part_name'].value_counts()
        parts_percentage = (parts_count / len(self.claims_data) * 100).round(2)
        
        fig_parts = go.Figure(data=[
            go.Bar(
                x=parts_count.head(10).index,
                y=parts_count.head(10).values,
                text=parts_percentage.head(10).apply(lambda x: f'{x}%'),
                textposition='outside'
            )
        ])
        fig_parts.update_layout(
            title="Top 10 Most Commonly Damaged Parts",
            xaxis_title="Part Name",
            yaxis_title="Number of Claims"
        )
        st.plotly_chart(fig_parts, use_container_width=True)
        
        # Parts Cost Analysis
        st.markdown("### Parts Cost Analysis")
        parts_df = pd.read_csv('data/Primary_Parts_Code.csv')
        
        fig_cost = px.box(
            parts_df,
            x='Category',
            y='Average_Cost',
            title='Cost Distribution by Part Category'
        )
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Parts Relationships
        st.markdown("### Parts Damage Patterns")
        
        # Create co-occurrence matrix
        all_parts = []
        for parts_list in self.claims_data['parts_affected'].apply(eval):
            all_parts.extend(list(itertools.combinations(sorted(parts_list), 2)))
        
        co_occurrence = pd.DataFrame(all_parts, columns=['Part1', 'Part2'])
        co_occurrence_counts = co_occurrence.groupby(['Part1', 'Part2']).size().reset_index(name='count')
        co_occurrence_counts = co_occurrence_counts.sort_values('count', ascending=False)
        
        # Create network graph
        fig_network = go.Figure()
        
        # Add edges (connections between parts)
        for _, row in co_occurrence_counts.head(20).iterrows():
            fig_network.add_trace(
                go.Scatter(
                    x=[row['Part1'], row['Part2']],
                    y=[0, 0],
                    mode='lines+markers',
                    line=dict(width=row['count']/2),
                    name=f"{row['Part1']} - {row['Part2']}"
                )
            )
        
        fig_network.update_layout(
            title="Part Damage Relationships (Top 20 Co-occurrences)",
            showlegend=False
        )
        st.plotly_chart(fig_network, use_container_width=True)

    def show_mapping_analysis(self):
        """Display AI mapping analysis"""
        st.subheader("Parts Mapping Analysis Dashboard")
        
        try:
            # Load data and convert timestamp
            garage_data = pd.read_csv('data/garage.csv')
            garage_data['mapping_timestamp'] = pd.to_datetime(garage_data['mapping_timestamp'])  # Convert to datetime
            surveyor_data = pd.read_csv('data/Primary_Parts_Code.csv')
            
            # Get real-time mapping stats from API
            try:
                response = requests.get(f"{self.api_base_url}/parts/mapping/stats", timeout=2)
                if response.status_code == 200:
                    mapping_stats = response.json()
                else:
                    mapping_stats = None
            except requests.exceptions.RequestException:
                mapping_stats = None

            # Show statistics
            st.markdown("### Mapping Performance Overview")
            col1, col2, col3 = st.columns(3)
            
            total_mappings = len(garage_data)
            successful = len(garage_data[garage_data['mapped_part_code'].notna()])
            
            with col1:
                st.metric(
                    "Total Mappings",
                    f"{total_mappings:,}",
                    help="Total number of parts requiring mapping"
                )
            with col2:
                st.metric(
                    "Successful Mappings",
                    f"{successful:,}",
                    f"{(successful/total_mappings*100):.1f}% success rate"
                )
            with col3:
                st.metric(
                    "Average Confidence",
                    f"{garage_data['mapping_confidence'].mean():.1f}%",
                    help="Average confidence score of AI mappings"
                )

            # Add Processing Time Metrics
            st.markdown("### Processing Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                avg_time = garage_data['processing_time'].mean()
                st.metric(
                    "Average Processing Time",
                    f"{avg_time:.2f} seconds",
                    f"{garage_data['processing_time'].std():.2f}s std dev"
                )
            with col2:
                success_rate = (successful / total_mappings) * 100
                st.metric(
                    "Overall Success Rate",
                    f"{success_rate:.1f}%",
                    f"{success_rate - 75:.1f}% vs target"
                )

            # Mapping Examples with Confidence Visualization
            st.markdown("### Recent Mapping Examples")
            
            # Get examples across confidence ranges
            examples = pd.concat([
                garage_data[garage_data['mapping_confidence'] >= 90].sample(2),  # High confidence
                garage_data[(garage_data['mapping_confidence'] >= 70) & 
                           (garage_data['mapping_confidence'] < 90)].sample(2),  # Medium confidence
                garage_data[garage_data['mapping_confidence'] < 70].sample(1)    # Low confidence
            ]).sort_values('mapping_confidence', ascending=False)
            
            for _, example in examples.iterrows():
                with st.expander(f"Mapping Example - Confidence: {example['mapping_confidence']:.1f}%"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Description:**")
                        st.write(example['part_description'])
                        st.markdown("**Category:**")
                        st.write(example['part_category'])
                    with col2:
                        st.markdown("**Mapped To:**")
                        st.write(example['mapped_part_name'] if pd.notna(example['mapped_part_name']) else "No mapping")
                        if pd.notna(example['error_type']):
                            st.markdown("**Error Type:**")
                            st.write(example['error_type'])
                        st.markdown("**Confidence Score:**")
                        st.progress(float(example['mapping_confidence']) / 100.0)
                        st.markdown(f"**Processing Time:** {example['processing_time']:.3f}s")

            # Mapping Performance Over Time
            st.markdown("### Mapping Performance Trends")
            
            # Daily performance metrics
            daily_metrics = garage_data.groupby(
                pd.Grouper(key='mapping_timestamp', freq='D')
            ).agg({
                'mapping_confidence': 'mean',
                'processing_time': 'mean',
                'mapped_part_code': lambda x: (x.notna().sum() / len(x)) * 100
            }).reset_index()
            
            # Create performance trend visualization
            fig = go.Figure()
            
            # Add confidence trend
            fig.add_trace(go.Scatter(
                x=daily_metrics['mapping_timestamp'],
                y=daily_metrics['mapping_confidence'],
                name='Confidence Score',
                line=dict(color='blue')
            ))
            
            # Add success rate trend
            fig.add_trace(go.Scatter(
                x=daily_metrics['mapping_timestamp'],
                y=daily_metrics['mapped_part_code'],
                name='Success Rate',
                line=dict(color='green')
            ))
            
            # Add processing time trend (on secondary y-axis)
            fig.add_trace(go.Scatter(
                x=daily_metrics['mapping_timestamp'],
                y=daily_metrics['processing_time'],
                name='Processing Time',
                line=dict(color='red', dash='dot'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Daily Mapping Performance Metrics',
                xaxis_title='Date',
                yaxis_title='Percentage (%)',
                yaxis2=dict(
                    title='Processing Time (s)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Error Analysis
            st.markdown("### Error Analysis")
            
            error_data = garage_data[pd.notna(garage_data['error_type'])]
            if not error_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Error type distribution
                    error_counts = error_data['error_type'].value_counts()
                    fig_errors = px.pie(
                        values=error_counts.values,
                        names=error_counts.index,
                        title='Distribution of Error Types'
                    )
                    st.plotly_chart(fig_errors, use_container_width=True)
                
                with col2:
                    # Error rates by category
                    category_errors = pd.crosstab(
                        garage_data['part_category'],
                        pd.notna(garage_data['error_type'])
                    ).apply(lambda x: (x[True] / (x[True] + x[False])) * 100 if True in x else 0, axis=1)
                    
                    fig_category = px.bar(
                        x=category_errors.index,
                        y=category_errors.values,
                        title='Error Rate by Category',
                        labels={'x': 'Category', 'y': 'Error Rate (%)'}
                    )
                    st.plotly_chart(fig_category, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load mapping analysis: {e}")
            logger.error(f"Mapping analysis error: {e}")

    def show_fraud_analytics(self):
        """Display fraud analytics page"""
        st.subheader("Fraud Analytics Dashboard")
        
        # Fraud Risk Distribution
        st.markdown("### Fraud Risk Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_risk = px.histogram(
                self.claims_data,
                x='fraud_risk',
                nbins=30,
                title='Distribution of Fraud Risk Scores'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            risk_categories = pd.cut(
                self.claims_data['fraud_risk'],
                bins=[0, 20, 40, 60, 80, 100],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            ).value_counts()
            
            fig_risk_pie = px.pie(
                values=risk_categories.values,
                names=risk_categories.index,
                title='Fraud Risk Categories'
            )
            st.plotly_chart(fig_risk_pie, use_container_width=True)
        
        # Risk vs Amount Analysis
        st.markdown("### Risk vs Amount Analysis")
        
        fig_scatter = px.scatter(
            self.claims_data,
            x='total_amount',
            y='fraud_risk',
            color='vehicle_type',
            size='processing_time',
            hover_data=['claim_id', 'status'],
            title='Fraud Risk vs Claim Amount'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Anomaly Detection
        st.markdown("### Anomaly Detection")
        
        # Calculate Z-scores for amount and risk
        z_amount = np.abs(stats.zscore(self.claims_data['total_amount']))
        z_risk = np.abs(stats.zscore(self.claims_data['fraud_risk']))
        
        # Mark anomalies
        anomalies = (z_amount > 2) | (z_risk > 2)
        
        fig_anomaly = go.Figure()
        
        # Add normal points
        fig_anomaly.add_trace(go.Scatter(
            x=self.claims_data[~anomalies]['total_amount'],
            y=self.claims_data[~anomalies]['fraud_risk'],
            mode='markers',
            name='Normal Claims',
            marker=dict(color='blue', size=8)
        ))
        
        # Add anomalies
        fig_anomaly.add_trace(go.Scatter(
            x=self.claims_data[anomalies]['total_amount'],
            y=self.claims_data[anomalies]['fraud_risk'],
            mode='markers',
            name='Potential Anomalies',
            marker=dict(color='red', size=12, symbol='x')
        ))
        
        fig_anomaly.update_layout(
            title='Anomaly Detection in Claims',
            xaxis_title='Claim Amount',
            yaxis_title='Fraud Risk Score'
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

    def show_trend_analysis(self):
        """Display trend analysis page"""
        st.subheader("Trend Analysis Dashboard")
        
        # Time period selector
        time_period = st.selectbox(
            "Select Time Period",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"]
        )
        
        # Filter data based on time period
        end_date = pd.Timestamp.now()
        if time_period == "Last 7 Days":
            start_date = end_date - pd.Timedelta(days=7)
        elif time_period == "Last 30 Days":
            start_date = end_date - pd.Timedelta(days=30)
        elif time_period == "Last 90 Days":
            start_date = end_date - pd.Timedelta(days=90)
        else:
            start_date = self.claims_data['timestamp'].min()
        
        filtered_data = self.claims_data[
            (self.claims_data['timestamp'] >= start_date) & 
            (self.claims_data['timestamp'] <= end_date)
        ]
        
        # Claims Volume Trend
        st.markdown("### Claims Volume Trend")
        
        daily_claims = filtered_data.groupby(
            pd.Grouper(key='timestamp', freq='D')
        ).agg({
            'claim_id': 'count',
            'total_amount': 'sum',
            'fraud_risk': 'mean'
        }).reset_index()
        
        fig_trend = go.Figure()
        
        # Add traces
        fig_trend.add_trace(go.Scatter(
            x=daily_claims['timestamp'],
            y=daily_claims['claim_id'],
            name='Number of Claims',
            line=dict(color='blue')
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=daily_claims['timestamp'],
            y=daily_claims['total_amount']/1000,
            name='Total Amount (K)',
            yaxis='y2',
            line=dict(color='green')
        ))
        
        fig_trend.update_layout(
            title='Claims Volume and Amount Trend',
            yaxis=dict(title='Number of Claims'),
            yaxis2=dict(title='Total Amount (K)', overlaying='y', side='right')
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Part Damage Trends
        st.markdown("### Part Damage Trends")
        
        # Calculate part damage frequency over time
        part_trends = []
        for _, row in filtered_data.iterrows():
            for part in eval(row['parts_affected']):
                part_trends.append({
                    'timestamp': row['timestamp'],
                    'part': part
                })
        
        part_trends_df = pd.DataFrame(part_trends)
        part_trends_df = part_trends_df.groupby(
            [pd.Grouper(key='timestamp', freq='W'), 'part']
        ).size().reset_index(name='count')
        
        # Show top 5 parts
        top_parts = part_trends_df.groupby('part')['count'].sum().nlargest(5).index
        
        fig_parts_trend = px.line(
            part_trends_df[part_trends_df['part'].isin(top_parts)],
            x='timestamp',
            y='count',
            color='part',
            title='Weekly Damage Frequency for Top 5 Parts'
        )
        
        st.plotly_chart(fig_parts_trend, use_container_width=True)

if __name__ == "__main__":
    dashboard = EnhancedDashboard()
    dashboard.run() 