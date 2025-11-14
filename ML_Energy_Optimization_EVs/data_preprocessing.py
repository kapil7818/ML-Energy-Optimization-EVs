import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic data for hybrid energy storage electric vehicles.

    Parameters:
    num_samples (int): Number of data points to generate.

    Returns:
    pd.DataFrame: Synthetic dataset with features and target.
    """
    np.random.seed(42)  # For reproducibility

    # Features: speed, acceleration, load, battery_soc, supercap_soc
    speed = np.random.uniform(0, 120, num_samples)  # km/h
    acceleration = np.random.uniform(-5, 5, num_samples)  # m/s²
    load = np.random.uniform(0, 1000, num_samples)  # kg
    battery_soc = np.random.uniform(20, 100, num_samples)  # %
    supercap_soc = np.random.uniform(0, 100, num_samples)  # %

    # Target: energy_consumption (kWh) - simulated based on features
    # Simple linear relationship with noise
    energy_consumption = (0.1 * speed + 0.05 * acceleration + 0.02 * load +
                          0.01 * (100 - battery_soc) + 0.005 * (100 - supercap_soc) +
                          np.random.normal(0, 0.5, num_samples))

    data = pd.DataFrame({
        'speed': speed,
        'acceleration': acceleration,
        'load': load,
        'battery_soc': battery_soc,
        'supercap_soc': supercap_soc,
        'energy_consumption': energy_consumption
    })

    return data

def preprocess_data(data, test_size=0.2, random_state=42):
    """
    Preprocess the data: split into train/test and scale features.

    Parameters:
    data (pd.DataFrame): Input dataset.
    test_size (float): Proportion of data for testing.
    random_state (int): Random state for reproducibility.

    Returns:
    tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    X = data.drop('energy_consumption', axis=1)
    y = data['energy_consumption']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
