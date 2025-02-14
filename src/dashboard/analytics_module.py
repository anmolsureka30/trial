import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List
import streamlit as st
from datetime import datetime, timedelta

class ClaimsAnalytics:
    def __init__(self, claims_data: pd.DataFrame):
        """Initialize analytics module"""
        self.claims_data = claims_data
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for analytics"""
        if not self.claims_data.empty:
            # Convert timestamp to datetime
            self.claims_data['timestamp'] = pd.to_datetime(self.claims_data['timestamp'])
            
            # Add derived features
            self.claims_data['month'] = self.claims_data['timestamp'].dt.strftime('%Y-%m')
            self.claims_data['weekday'] = self.claims_data['timestamp'].dt.day_name()
            self.claims_data['hour'] = self.claims_data['timestamp'].dt.hour

    def show_trend_analysis(self):
        """Show claims trends analysis"""
        st.subheader("Claims Trends Analysis")
        
        # Monthly trends
        monthly_claims = self.claims_data.groupby('month').agg({
            'claim_id': 'count',
            'total_amount': 'sum',
            'fraud_risk': 'mean'
        }).reset_index()
        
        # Create subplot with 3 metrics
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Claims Count", "Total Amount", "Average Fraud Risk"),
            vertical_spacing=0.1
        )
        
        # Claims count trend
        fig.add_trace(
            go.Scatter(x=monthly_claims['month'], y=monthly_claims['claim_id'],
                      mode='lines+markers', name='Claims Count'),
            row=1, col=1
        )
        
        # Amount trend
        fig.add_trace(
            go.Scatter(x=monthly_claims['month'], y=monthly_claims['total_amount'],
                      mode='lines+markers', name='Total Amount'),
            row=2, col=1
        )
        
        # Fraud risk trend
        fig.add_trace(
            go.Scatter(x=monthly_claims['month'], y=monthly_claims['fraud_risk'],
                      mode='lines+markers', name='Fraud Risk'),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def show_pattern_analysis(self):
        """Show claims pattern analysis"""
        st.subheader("Claims Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weekday distribution
            weekday_claims = self.claims_data['weekday'].value_counts()
            fig = px.bar(
                x=weekday_claims.index,
                y=weekday_claims.values,
                title="Claims by Day of Week"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hourly distribution
            hourly_claims = self.claims_data['hour'].value_counts().sort_index()
            fig = px.line(
                x=hourly_claims.index,
                y=hourly_claims.values,
                title="Claims by Hour of Day"
            )
            st.plotly_chart(fig, use_container_width=True)

    def show_fraud_analysis(self):
        """Show fraud analysis"""
        st.subheader("Fraud Analysis")
        
        # Fraud risk distribution
        fig = px.histogram(
            self.claims_data,
            x='fraud_risk',
            nbins=30,
            title="Distribution of Fraud Risk Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud risk by amount
        fig = px.scatter(
            self.claims_data,
            x='total_amount',
            y='fraud_risk',
            title="Fraud Risk vs Claim Amount",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_cost_analysis(self):
        """Show cost analysis"""
        st.subheader("Cost Analysis")
        
        # Calculate key metrics
        metrics = {
            'Total Claims': len(self.claims_data),
            'Total Amount': self.claims_data['total_amount'].sum(),
            'Average Claim': self.claims_data['total_amount'].mean(),
            'Median Claim': self.claims_data['total_amount'].median(),
            'Max Claim': self.claims_data['total_amount'].max(),
            'Min Claim': self.claims_data['total_amount'].min()
        }
        
        # Display metrics
        cols = st.columns(3)
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i % 3]:
                st.metric(
                    metric,
                    f"${value:,.2f}" if 'Amount' in metric or 'Claim' in metric else value
                )
        
        # Cost distribution
        fig = px.box(
            self.claims_data,
            y='total_amount',
            title="Claim Amount Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_predictive_insights(self):
        """Show predictive insights"""
        st.subheader("Predictive Insights")
        
        # Calculate trends
        monthly_avg = self.claims_data.groupby('month')['total_amount'].mean()
        
        # Simple linear projection
        X = np.arange(len(monthly_avg)).reshape(-1, 1)
        y = monthly_avg.values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Project next 3 months
        future_months = pd.date_range(
            start=monthly_avg.index[-1],
            periods=4,
            freq='M'
        ).strftime('%Y-%m')
        
        X_future = np.arange(len(monthly_avg), len(monthly_avg) + 3).reshape(-1, 1)
        y_future = model.predict(X_future)
        
        # Plot actual vs predicted
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=monthly_avg.index,
            y=monthly_avg.values,
            name="Actual",
            mode="lines+markers"
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=future_months[1:],
            y=y_future,
            name="Predicted",
            mode="lines+markers",
            line=dict(dash='dash')
        ))
        
        fig.update_layout(title="Claims Amount Trend and Projection")
        st.plotly_chart(fig, use_container_width=True) 