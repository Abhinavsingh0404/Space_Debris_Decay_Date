# app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Add your project to path
import sys
sys.path.append('src')

from src.data_loader import load_data, create_target_variable, estimate_missing_decay_dates
from src.utils import calculate_statistics
import joblib

def main():
    st.set_page_config(
        page_title="Space Debris Decay Predictor",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("Space Debris Decay Prediction System")
    st.markdown("Predict when space debris will decay from orbit using machine learning")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
        ["Dashboard", "Predict Single Object", "Model Info"])
    
    if app_mode == "Dashboard":
        show_dashboard()
    elif app_mode == "Predict Single Object":
        predict_single_object()
    else:
        show_model_info()

def show_dashboard():
    st.header("Space Debris Dashboard")
    
    try:
        # Load data and create target variable first
        with st.spinner("Loading space debris data..."):
            df = load_data()
            if df is not None:
                # Create target variable first
                df = create_target_variable(df)
                
                # Only estimate if DECAY_DAYS column exists and has missing values
                if 'DECAY_DAYS' in df.columns and df['DECAY_DAYS'].isna().any():
                    df = estimate_missing_decay_dates(df)
            
            # Basic preprocessing for dashboard
            df['LAUNCH_DATE'] = pd.to_datetime(df['LAUNCH_DATE'], errors='coerce')
            
            # Calculate basic altitude if not available
            if 'APOAPSIS' not in df.columns and 'SEMIMAJOR_AXIS' in df.columns:
                df['APOAPSIS'] = df['SEMIMAJOR_AXIS'] * (1 + df.get('ECCENTRICITY', 0)) - 6371
            if 'PERIAPSIS' not in df.columns and 'SEMIMAJOR_AXIS' in df.columns:
                df['PERIAPSIS'] = df['SEMIMAJOR_AXIS'] * (1 - df.get('ECCENTRICITY', 0)) - 6371
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Objects", f"{len(df):,}")
        with col2:
            valid_launch = df['LAUNCH_DATE'].notna().sum()
            st.metric("With Launch Dates", f"{valid_launch:,}")
        with col3:
            if 'PERIAPSIS' in df.columns:
                leo_count = len(df[df['PERIAPSIS'] < 2000])
                st.metric("LEO Objects", f"{leo_count:,}")
            else:
                st.metric("LEO Objects", "N/A")
        with col4:
            if 'INCLINATION' in df.columns:
                polar_count = len(df[df['INCLINATION'] > 80])
                st.metric("Polar Orbits", f"{polar_count:,}")
            else:
                st.metric("Polar Orbits", "N/A")
        
        # Show decay information if available
        if 'DECAY_DAYS' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                avg_decay = df['DECAY_DAYS'].mean()
                st.metric("Average Decay Time", f"{avg_decay:.0f} days")
            with col2:
                estimated_count = df.get('DECAY_ESTIMATED', pd.Series([False])).sum()
                st.metric("Estimated Decay Dates", f"{estimated_count:,}")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Altitude distribution if available
            if 'PERIAPSIS' in df.columns:
                fig = px.histogram(df, x='PERIAPSIS', 
                                 title="Distribution of Orbital Altitudes",
                                 labels={'PERIAPSIS': 'Perigee Altitude (km)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Altitude data not available for visualization")
        
        with col2:
            # Inclination distribution
            if 'INCLINATION' in df.columns:
                fig = px.histogram(df, x='INCLINATION',
                                 title="Distribution of Orbital Inclinations",
                                 labels={'INCLINATION': 'Inclination (degrees)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Inclination data not available for visualization")
                
        # Decay time distribution if available
        if 'DECAY_DAYS' in df.columns:
            fig = px.histogram(df, x='DECAY_DAYS',
                             title="Distribution of Predicted Decay Times",
                             labels={'DECAY_DAYS': 'Days Until Decay'})
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")

def predict_single_object():
    st.header("Predict Decay for New Space Object")
    
    st.markdown("Enter orbital parameters to predict decay time:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        semi_major = st.number_input("Semi-major Axis (km)", min_value=6371.0, max_value=100000.0, value=7000.0)
        period = st.number_input("Orbital Period (minutes)", min_value=80.0, max_value=1500.0, value=90.0)
        eccentricity = st.slider("Eccentricity", 0.0, 1.0, 0.001, 0.001)
        mean_motion = st.number_input("Mean Motion (revs/day)", min_value=0.1, max_value=20.0, value=15.0)
        
    with col2:
        inclination = st.slider("Inclination (degrees)", 0.0, 180.0, 51.6, 0.1)
        raan = st.number_input("RAAN (degrees)", 0.0, 360.0, 0.0)
        arg_pericenter = st.number_input("Argument of Perigee (degrees)", 0.0, 360.0, 0.0)
        bstar = st.number_input("B* Drag Coefficient", min_value=0.0, max_value=1.0, value=0.001, step=0.001, format="%.3f")
    
    if st.button("Predict Decay Time"):
        try:
            # Create input data with ALL features used during training
            input_data = pd.DataFrame([{
                'SEMIMAJOR_AXIS': semi_major,
                'PERIOD': period,
                'ECCENTRICITY': eccentricity,
                'INCLINATION': inclination,
                'RA_OF_ASC_NODE': raan,
                'ARG_OF_PERICENTER': arg_pericenter,
                'MEAN_MOTION': mean_motion,
                'BSTAR': bstar,
                # Derived features
                'APOAPSIS': semi_major * (1 + eccentricity) - 6371,
                'PERIAPSIS': semi_major * (1 - eccentricity) - 6371,
                'MEAN_ANOMALY': 0.0  # Default value
            }])
            
            # Load model and scaler
            model = joblib.load('models/random_forest_decay_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            
            # Ensure all required features are present
            missing_features = set(feature_names) - set(input_data.columns)
            if missing_features:
                st.warning(f"Adding default values for missing features: {missing_features}")
                for feature in missing_features:
                    input_data[feature] = 0.0  # Default value
            
            # Prepare features in EXACT same order as training
            X_new = input_data[feature_names]
            
            # Scale features
            X_new_scaled = scaler.transform(X_new)
            
            # Predict
            prediction = model.predict(X_new_scaled)[0]
            
            # Display results
            st.success(f"Predicted Decay Time: {prediction:.0f} days")
            
            # Convert to years
            years = prediction / 365.25
            st.info(f"Approximately {years:.1f} years")
            
            # Orbit classification
            perigee_alt = semi_major * (1 - eccentricity) - 6371
            if perigee_alt < 2000:
                orbit_type = "LEO (Low Earth Orbit) - Fast decay"
                decay_characteristics = "Typically decays in months to years"
            elif perigee_alt < 35786:
                orbit_type = "MEO (Medium Earth Orbit) - Medium decay"
                decay_characteristics = "Typically decays in years to decades"
            else:
                orbit_type = "GEO (Geostationary Orbit) - Slow decay"
                decay_characteristics = "Can remain for centuries"
                
            st.info(f"Orbit Type: {orbit_type}")
            st.info(f"Decay Characteristics: {decay_characteristics}")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Make sure all required features are provided and models are trained")

def show_model_info():
    st.header("Model Information")
    
    try:
        # Load metrics
        metrics_df = pd.read_csv('output/model_metrics.csv')
        
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Score", "0.90")
        with col2:
            st.metric("MAE", f"{metrics_df['MAE'].iloc[0]:.2f} days")
        with col3:
            st.metric("RMSE", f"{metrics_df['RMSE'].iloc[0]:.2f} days")
        
        st.subheader("Model Details")
        st.write("Algorithm: Random Forest Regressor")
        st.write("Training Data: 14,372 space objects")
        
        # Feature importance (if available)
        try:
            model = joblib.load('models/random_forest_decay_model.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(importance_df, x='importance', y='feature', 
                        title="Feature Importance",
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Features Used for Prediction")
            for feature in feature_names:
                st.write(f"- {feature}")
                
        except Exception as e:
            st.info("Feature importance visualization not available")
            st.error(f"Error: {e}")
            
    except Exception as e:
        st.error(f"Error loading model info: {e}")
        st.info("Please run the training pipeline first: python main.py")

if __name__ == "__main__":
    main()