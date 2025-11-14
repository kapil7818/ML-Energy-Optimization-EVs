import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_preprocessing import generate_synthetic_data, preprocess_data


class TestDataPreprocessing:
    """Test cases for data preprocessing module."""

    def test_generate_synthetic_data_basic(self):
        """Test basic functionality of synthetic data generation."""
        data = generate_synthetic_data(num_samples=100)

        # Check if data is DataFrame
        assert isinstance(data, pd.DataFrame)

        # Check expected columns
        expected_columns = ['speed', 'acceleration', 'load', 'battery_soc', 'supercap_soc', 'energy_consumption']
        assert list(data.columns) == expected_columns

        # Check data shape
        assert data.shape == (100, 6)

    def test_generate_synthetic_data_ranges(self):
        """Test that generated data is within expected ranges."""
        data = generate_synthetic_data(num_samples=1000)

        # Check ranges for each feature
        assert data['speed'].min() >= 0 and data['speed'].max() <= 120
        assert data['acceleration'].min() >= -5 and data['acceleration'].max() <= 5
        assert data['load'].min() >= 0 and data['load'].max() <= 1000
        assert data['battery_soc'].min() >= 20 and data['battery_soc'].max() <= 100
        assert data['supercap_soc'].min() >= 0 and data['supercap_soc'].max() <= 100

    def test_generate_synthetic_data_reproducibility(self):
        """Test that data generation is reproducible with same seed."""
        data1 = generate_synthetic_data(num_samples=50)
        data2 = generate_synthetic_data(num_samples=50)

        # Should be identical due to fixed seed
        pd.testing.assert_frame_equal(data1, data2)

    def test_preprocess_data_output_types(self):
        """Test that preprocess_data returns correct types."""
        data = generate_synthetic_data(num_samples=100)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

        # Check types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(scaler, StandardScaler)

    def test_preprocess_data_split_ratios(self):
        """Test that data is split correctly."""
        data = generate_synthetic_data(num_samples=100)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data, test_size=0.2)

        # Check split ratios
        assert len(X_train) == 80  # 80% of 100
        assert len(X_test) == 20   # 20% of 100
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_preprocess_data_scaling(self):
        """Test that features are properly scaled."""
        data = generate_synthetic_data(num_samples=100)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

        # Check that training data is scaled (mean close to 0, std close to 1)
        assert abs(X_train.mean()) < 0.1  # Should be close to 0
        assert abs(X_train.std() - 1) < 0.1  # Should be close to 1

    def test_preprocess_data_no_data_leakage(self):
        """Test that there's no data leakage between train and test sets."""
        data = generate_synthetic_data(num_samples=100)
        data['unique_id'] = range(100)  # Add unique identifier

        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

        # Get original indices
        train_indices = data.index[:80]  # First 80 for training
        test_indices = data.index[80:]   # Last 20 for testing

        # Check that train and test indices don't overlap
        assert len(set(train_indices) & set(test_indices)) == 0

    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3])
    def test_preprocess_data_different_test_sizes(self, test_size):
        """Test preprocessing with different test sizes."""
        data = generate_synthetic_data(num_samples=100)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data, test_size=test_size)

        expected_train_size = int(100 * (1 - test_size))
        expected_test_size = 100 - expected_train_size

        assert len(X_train) == expected_train_size
        assert len(X_test) == expected_test_size
