import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from logging_config import get_logger, log_exceptions
import traceback

# Set up logging
logger = get_logger(__name__)

@st.cache_resource
@log_exceptions
def load_model_and_scaler():
    """Load the trained ML model and scaler."""
    try:
        model = joblib.load('optimized_random_forest_model.pkl')
        # Try to load saved scaler, fallback to new one
        try:
            scaler = joblib.load('scaler.pkl')
        except FileNotFoundError:
            scaler = StandardScaler()
            logger.warning("Scaler file not found, using new scaler")
        logger.info("Model and scaler loaded successfully")
        return model, scaler
    except FileNotFoundError:
        logger.error("Model file 'optimized_random_forest_model.pkl' not found")
        st.error("Model file not found. Please run the training pipeline first.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading model/scaler: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None, None

@log_exceptions
def make_prediction(model, scaler, input_data):
    """Make energy consumption prediction."""
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        logger.info(".2f")
        return prediction
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

def plot_feature_importance(model, feature_names):
    """Create feature importance plot."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(importances)), importances[indices], color='skyblue', alpha=0.8)
            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            # Add value labels
            for bar, imp in zip(bars, importances[indices]):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       '.3f', ha='left', va='center', fontsize=10)

            return fig
        else:
            logger.warning("Model does not support feature importance")
            return None
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        return None

def main():
    """Main Streamlit application."""
    try:
        # Page configuration
        st.set_page_config(
            page_title="EV Energy Predictor",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🚗 Advanced Energy Consumption Predictor for Hybrid EVs")
        st.markdown("*Predicting energy consumption in hybrid energy storage electric vehicles using optimized machine learning models.*")

        # Load model and scaler
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            return

        # Sidebar for inputs
        st.sidebar.header("🔧 Vehicle Parameters Input")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            speed = st.slider("Speed (km/h)", 0.0, 120.0, 60.0, 0.1, help="Vehicle speed in kilometers per hour")
            acceleration = st.slider("Acceleration (m/s²)", -5.0, 5.0, 0.0, 0.1,
                                   help="Acceleration in meters per second squared")

        with col2:
            load = st.slider("Load (kg)", 0.0, 1000.0, 500.0, 1.0, help="Vehicle load in kilograms")
            battery_soc = st.slider("Battery SOC (%)", 20.0, 100.0, 80.0, 0.1,
                                  help="Battery State of Charge percentage")
            supercap_soc = st.slider("Supercapacitor SOC (%)", 0.0, 100.0, 50.0, 0.1,
                                   help="Supercapacitor State of Charge percentage")

        # Create input dataframe
        input_data = pd.DataFrame({
            'speed': [speed],
            'acceleration': [acceleration],
            'load': [load],
            'battery_soc': [battery_soc],
            'supercap_soc': [supercap_soc]
        })

        # Main content area
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("📊 Prediction Results")

            # Make prediction
            prediction = make_prediction(model, scaler, input_data.values)

            # Display prediction with formatting
            st.success(f"**Predicted Energy Consumption:** {prediction:.3f} kWh")

            # Prediction confidence indicator
            if prediction < 5:
                st.info("🟢 Low energy consumption predicted")
            elif prediction < 10:
                st.warning("🟡 Moderate energy consumption predicted")
            else:
                st.error("🔴 High energy consumption predicted")

            # Display input parameters
            st.subheader("Input Parameters")
            st.dataframe(input_data)

        with col2:
            st.header("📈 Feature Importance")

            feature_names = list(input_data.columns)
            fig = plot_feature_importance(model, feature_names)

            if fig is not None:
                st.pyplot(fig)
            else:
                st.warning("Feature importance not available for this model type")

        # Additional analysis section
        st.header("🔍 Detailed Analysis")

        tab1, tab2, tab3 = st.tabs(["Parameter Impact", "Model Info", "About"])

        with tab1:
            st.subheader("How Parameters Affect Energy Consumption")

            # Create sample predictions for different parameter values
            base_params = [60, 0, 500, 80, 50]  # baseline values

            # Speed variation
            speeds = np.linspace(0, 120, 13)
            speed_predictions = []
            for s in speeds:
                params = base_params.copy()
                params[0] = s
                pred = make_prediction(model, scaler, np.array([params]))
                speed_predictions.append(pred)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(speeds, speed_predictions, 'b-o', linewidth=2, markersize=4)
            ax.set_xlabel('Speed (km/h)')
            ax.set_ylabel('Predicted Energy Consumption (kWh)')
            ax.set_title('Energy Consumption vs Speed')
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with tab2:
            st.subheader("Model Information")
            st.markdown("""
            **Model Type:** Optimized Random Forest Regressor
            **Training Data:** 1000 synthetic samples
            **Features:** 5 vehicle parameters
            **Optimization:** Hyperparameter tuning with Grid Search
            **Cross-Validation:** 3-fold CV
            **Performance:** R² > 0.99, MSE < 0.5
            """)

        with tab3:
            st.subheader("About This Application")
            st.markdown("""
            This advanced application uses machine learning to predict energy consumption
            in hybrid energy storage electric vehicles (HESS EVs).

            **Key Features:**
            - Real-time energy consumption prediction
            - Feature importance analysis
            - Parameter impact visualization
            - Optimized Random Forest model with hyperparameter tuning

            **Parameters Considered:**
            - **Speed:** Vehicle velocity affects aerodynamic drag
            - **Acceleration:** Rapid changes consume more energy
            - **Load:** Additional weight increases energy requirements
            - **Battery SOC:** Lower charge levels may affect efficiency
            - **Supercapacitor SOC:** High-power storage state

            **Technology Stack:**
            - Machine Learning: Scikit-learn, Random Forest
            - Web Framework: Streamlit
            - Data Processing: Pandas, NumPy
            - Visualization: Matplotlib, Seaborn
            - Logging: Python logging module
            """)

            st.markdown("---")
            st.markdown("*Built as part of B.Tech CSE final year project*")

        # Footer
        st.markdown("---")
        st.markdown("*© 2024 EV Energy Optimization Project*")

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please check the logs for more details.")

if __name__ == "__main__":
    main()
