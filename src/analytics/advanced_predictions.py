import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedPredictions:
    def __init__(self, claims_data: pd.DataFrame):
        """Initialize advanced predictions module"""
        self.claims_data = claims_data
        self.prepare_data()
        self.setup_models()

    def prepare_data(self):
        """Prepare data for predictions"""
        try:
            if not self.claims_data.empty:
                # Ensure datetime
                self.claims_data['timestamp'] = pd.to_datetime(self.claims_data['timestamp'])
                
                # Create time-based features
                self.claims_data['year'] = self.claims_data['timestamp'].dt.year
                self.claims_data['month'] = self.claims_data['timestamp'].dt.month
                self.claims_data['day'] = self.claims_data['timestamp'].dt.day
                self.claims_data['dayofweek'] = self.claims_data['timestamp'].dt.dayofweek
                
                # Calculate rolling statistics
                self.claims_data['rolling_mean_amount'] = self.claims_data['total_amount'].rolling(window=7).mean()
                self.claims_data['rolling_std_amount'] = self.claims_data['total_amount'].rolling(window=7).std()
                
                logger.info("Data preparation completed successfully")
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise

    def setup_models(self):
        """Initialize prediction models"""
        try:
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7
            )
            
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            raise

    def generate_predictions(self) -> Dict:
        """Generate comprehensive predictions"""
        try:
            results = {}
            
            # Time series predictions
            results['time_series'] = self._prophet_predictions()
            
            # Amount predictions
            results['amount_predictions'] = self._amount_predictions()
            
            # Anomaly detection
            results['anomalies'] = self._detect_anomalies()
            
            # Risk predictions
            results['risk_predictions'] = self._predict_risk()
            
            return results
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return {}

    def _prophet_predictions(self) -> Dict:
        """Generate Prophet predictions"""
        try:
            # Prepare data for Prophet
            prophet_data = self.claims_data[['timestamp', 'total_amount']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Fit model
            self.prophet_model.fit(prophet_data)
            
            # Make future predictions
            future_dates = self.prophet_model.make_future_dataframe(periods=30)
            forecast = self.prophet_model.predict(future_dates)
            
            return {
                'dates': forecast['ds'].tail(30),
                'predictions': forecast['yhat'].tail(30),
                'lower_bound': forecast['yhat_lower'].tail(30),
                'upper_bound': forecast['yhat_upper'].tail(30)
            }
        except Exception as e:
            logger.error(f"Prophet predictions failed: {e}")
            return {}

    def _amount_predictions(self) -> Dict:
        """Generate amount predictions using ensemble"""
        try:
            # Prepare features
            features = ['year', 'month', 'day', 'dayofweek', 
                       'rolling_mean_amount', 'rolling_std_amount']
            X = self.claims_data[features].dropna()
            y = self.claims_data['total_amount'].loc[X.index]
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train models
            self.rf_model.fit(X_train, y_train)
            self.xgb_model.fit(X_train, y_train)
            
            # Generate predictions
            rf_pred = self.rf_model.predict(X_test)
            xgb_pred = self.xgb_model.predict(X_test)
            
            # Ensemble predictions
            ensemble_pred = (rf_pred + xgb_pred) / 2
            
            return {
                'actual': y_test.values,
                'predictions': ensemble_pred,
                'dates': self.claims_data['timestamp'].loc[X_test.index],
                'rf_importance': dict(zip(features, self.rf_model.feature_importances_))
            }
        except Exception as e:
            logger.error(f"Amount predictions failed: {e}")
            return {}

    def _detect_anomalies(self) -> Dict:
        """Detect anomalies in claims"""
        try:
            # Prepare features for anomaly detection
            features = ['total_amount', 'fraud_risk', 'rolling_mean_amount']
            X = self.claims_data[features].dropna()
            
            # Fit and predict
            anomaly_scores = self.anomaly_detector.fit_predict(X)
            
            # Identify anomalies
            anomalies = X[anomaly_scores == -1]
            anomaly_dates = self.claims_data['timestamp'].loc[anomalies.index]
            
            return {
                'anomaly_indices': anomalies.index.tolist(),
                'anomaly_dates': anomaly_dates.tolist(),
                'anomaly_values': anomalies['total_amount'].tolist()
            }
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {}

    def _predict_risk(self) -> Dict:
        """Predict risk levels for future claims"""
        try:
            # Prepare features for risk prediction
            features = ['total_amount', 'rolling_mean_amount', 'rolling_std_amount']
            X = self.claims_data[features].dropna()
            
            # Train a simple risk model
            risk_model = RandomForestRegressor(n_estimators=50, random_state=42)
            risk_model.fit(X, self.claims_data['fraud_risk'].loc[X.index])
            
            # Make predictions
            predictions = risk_model.predict(X)
            
            return {
                'predictions': predictions.tolist(),
                'dates': self.claims_data['timestamp'].loc[X.index].tolist(),
                'average_risk': float(np.mean(predictions))
            }
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return {}

    def show_predictions(self):
        """Display comprehensive prediction analysis"""
        try:
            predictions = self.generate_predictions()
            
            # Time series predictions
            st.subheader("Claims Amount Forecast")
            self._plot_time_series_forecast(predictions['time_series'])
            
            # Amount predictions
            st.subheader("Amount Prediction Analysis")
            self._plot_amount_predictions(predictions['amount_predictions'])
            
            # Anomaly detection
            st.subheader("Anomaly Detection")
            self._plot_anomalies(predictions['anomalies'])
            
            # Feature importance
            if 'rf_importance' in predictions['amount_predictions']:
                st.subheader("Feature Importance")
                self._plot_feature_importance(predictions['amount_predictions']['rf_importance'])
                
        except Exception as e:
            logger.error(f"Prediction visualization failed: {e}")
            st.error("Failed to display predictions. Please check logs.")

    def _plot_time_series_forecast(self, forecast: Dict):
        """Plot time series forecast"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=self.claims_data['timestamp'],
            y=self.claims_data['total_amount'],
            name="Historical",
            mode="lines"
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=forecast['dates'],
            y=forecast['predictions'],
            name="Forecast",
            mode="lines",
            line=dict(dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['dates'],
            y=forecast['upper_bound'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['dates'],
            y=forecast['lower_bound'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name="95% Confidence"
        ))
        
        st.plotly_chart(fig, use_container_width=True)

    def _plot_amount_predictions(self, predictions: Dict):
        """Plot amount predictions"""
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=predictions['dates'],
            y=predictions['actual'],
            name="Actual",
            mode="markers"
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=predictions['dates'],
            y=predictions['predictions'],
            name="Predicted",
            mode="lines"
        ))
        
        st.plotly_chart(fig, use_container_width=True)

    def _plot_anomalies(self, anomalies: Dict):
        """Plot detected anomalies"""
        fig = go.Figure()
        
        # All points
        fig.add_trace(go.Scatter(
            x=self.claims_data['timestamp'],
            y=self.claims_data['total_amount'],
            mode='markers',
            name='Normal Claims'
        ))
        
        # Anomalies
        fig.add_trace(go.Scatter(
            x=anomalies['anomaly_dates'],
            y=anomalies['anomaly_values'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Anomalies'
        ))
        
        st.plotly_chart(fig, use_container_width=True)

    def _plot_feature_importance(self, importance: Dict):
        """Plot feature importance"""
        fig = px.bar(
            x=list(importance.keys()),
            y=list(importance.values()),
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True) 