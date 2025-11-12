# src/pipeline.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split

def run_full_pipeline():
    """
    Run the complete space debris decay prediction pipeline
    """
    print("Starting Space Debris Decay Prediction Pipeline")
    print("=" * 60)
    
    try:
        # Import inside function to avoid circular imports
        from src.data_loader import load_data, create_target_variable, estimate_missing_decay_dates
        from src.preprocessing import handle_missing_values, engineer_features, prepare_features, scale_features
        from src.model_training import compare_models, train_final_model, evaluate_model, plot_model_comparison, save_plot
        
        # Step 1: Load and prepare data
        print("\nSTEP 1: Data Loading")
        df = load_data()
        if df is None:
            print("Failed to load data. Please check if space_decay.csv exists in data/ folder")
            return None
        
        df = create_target_variable(df)
        
        # Estimate missing decay dates if needed
        if df['DECAY_DAYS'].isna().all() or df['DECAY_DAYS'].notna().sum() == 0:
            print("No valid decay dates found. Estimating...")
            df = estimate_missing_decay_dates(df)
        
        # Step 2: Preprocessing and feature engineering
        print("\nSTEP 2: Preprocessing")
        numeric_features = ['SEMIMAJOR_AXIS', 'PERIOD', 'MEAN_MOTION', 'BSTAR', 'APOAPSIS', 'PERIAPSIS', 'INCLINATION', 'ECCENTRICITY']
        df = handle_missing_values(df, numeric_features)
        df = engineer_features(df)
        
        # Step 3: Prepare features for modeling
        print("\nSTEP 3: Feature Preparation")
        X, y, feature_names = prepare_features(df, numeric_features)
        
        # Step 4: Train-test split and scaling
       # Step 4: Train-test split and scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Ensure no NaN values in target (FINAL SAFETY CHECK)
        print(f"NaN values in y_train: {y_train.isna().sum()}")
        print(f"NaN values in y_test: {y_test.isna().sum()}")

        # Fill any remaining NaN values with median
        if y_train.isna().sum() > 0:
            y_train = y_train.fillna(y_train.median())
            print(f"Filled {y_train.isna().sum()} NaN values in y_train")
        if y_test.isna().sum() > 0:
            y_test = y_test.fillna(y_test.median())
            print(f"Filled {y_test.isna().sum()} NaN values in y_test")

        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        print(f"Training set: {X_train_scaled.shape}")
        print(f"Testing set:  {X_test_scaled.shape}")
        
        # Step 5: Model comparison
        print("\nSTEP 4: Model Comparison")
        results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Plot model comparison
        plot_model_comparison(results, 'model_comparison.png')
        
        # Step 6: Train final model
        print("\nSTEP 5: Final Model Training")
        final_model = train_final_model(X_train_scaled, y_train)
        final_metrics = evaluate_model(final_model, X_test_scaled, y_test, "Optimized Random Forest")
        
        # Step 7: Save the pipeline
        print("\nSTEP 6: Saving Pipeline")
        save_pipeline(final_model, scaler, feature_names, final_metrics)
        
        # Step 8: Plot predictions
        plot_predictions(y_test, final_metrics['predictions'])
        
        print("\nPipeline completed successfully!")
        return {
            'model': final_model,
            'scaler': scaler,
            'features': feature_names,
            'metrics': final_metrics
        }
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_pipeline(model, scaler, feature_names, metrics):
    """
    Save the trained model and preprocessing objects
    """
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)
    
    # Save model
    model_path = os.path.join('models', 'random_forest_decay_model.pkl')
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = os.path.join('models', 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save feature names
    features_path = os.path.join('models', 'feature_names.pkl')
    joblib.dump(feature_names, features_path)
    
    # Save metrics
    metrics_path = os.path.join('output', 'model_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    
    print(f"Model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Features saved: {features_path}")
    print(f"Metrics saved: {metrics_path}")

def plot_predictions(y_test, y_pred):
    """
    Plot actual vs predicted values
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Decay Days')
    plt.ylabel('Predicted Decay Days')
    plt.title('Actual vs Predicted Decay Days')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs('output/plots', exist_ok=True)
    plt.savefig('output/plots/predictions_plot.png', bbox_inches='tight', dpi=300)
    plt.show()

def predict_new_data(model, scaler, features, new_data):
    """
    Make predictions on new data
    """
    # Ensure new_data has the same features
    available_features = [f for f in features if f in new_data.columns]
    X_new = new_data[available_features]
    
    # Scale the features
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new_scaled)
    
    return predictions