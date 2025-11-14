import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from model_training import (
    train_linear_regression,
    train_random_forest,
    evaluate_model,
    save_model,
    load_model
)
import tempfile
import os


class TestModelTraining:
    """Test cases for model training module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        X = np.random.rand(100, 5)  # 5 features
        y = X.sum(axis=1) + np.random.normal(0, 0.1, 100)  # Linear relationship with noise
        return X, y

    def test_train_linear_regression(self, sample_data):
        """Test Linear Regression training."""
        X, y = sample_data
        model = train_linear_regression(X, y)

        assert isinstance(model, LinearRegression)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')

    def test_train_random_forest(self, sample_data):
        """Test Random Forest training."""
        X, y = sample_data
        model = train_random_forest(X, y)

        assert isinstance(model, RandomForestRegressor)
        assert hasattr(model, 'feature_importances_')
        assert model.n_estimators == 100  # Default value

    def test_train_random_forest_custom_params(self, sample_data):
        """Test Random Forest with custom parameters."""
        X, y = sample_data
        model = train_random_forest(X, y, n_estimators=50, random_state=123)

        assert model.n_estimators == 50
        assert model.random_state == 123

    def test_evaluate_model(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        model = train_linear_regression(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Evaluate
        metrics = evaluate_model(model, X, y)

        assert 'MSE' in metrics
        assert 'R2' in metrics
        assert isinstance(metrics['MSE'], float)
        assert isinstance(metrics['R2'], float)
        assert metrics['MSE'] >= 0  # MSE should be non-negative
        assert metrics['R2'] <= 1    # R² should be <= 1

    def test_evaluate_model_perfect_fit(self):
        """Test evaluation with perfect fit."""
        X = np.random.rand(50, 3)
        y = X.sum(axis=1)  # Perfect linear relationship

        model = train_linear_regression(X, y)
        metrics = evaluate_model(model, X, y)

        # Should have very low MSE and high R²
        assert metrics['MSE'] < 0.01
        assert metrics['R2'] > 0.99

    def test_save_and_load_model(self, sample_data):
        """Test saving and loading model."""
        X, y = sample_data
        model = train_random_forest(X, y)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            save_model(model, temp_path)
            assert os.path.exists(temp_path)

            # Load model
            loaded_model = load_model(temp_path)
            assert isinstance(loaded_model, RandomForestRegressor)

            # Test that loaded model works
            y_pred = loaded_model.predict(X)
            assert len(y_pred) == len(y)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_model_predictions_consistent(self, sample_data):
        """Test that model predictions are consistent."""
        X, y = sample_data
        model = train_random_forest(X, y, random_state=42)

        # Make predictions twice
        y_pred1 = model.predict(X)
        y_pred2 = model.predict(X)

        # Should be identical
        np.testing.assert_array_equal(y_pred1, y_pred2)

    def test_model_handles_different_input_shapes(self, sample_data):
        """Test model with different input shapes."""
        X, y = sample_data
        model = train_linear_regression(X, y)

        # Test with single sample
        single_sample = X[0:1]
        y_pred_single = model.predict(single_sample)
        assert y_pred_single.shape == (1,)

        # Test with multiple samples
        multi_samples = X[:5]
        y_pred_multi = model.predict(multi_samples)
        assert y_pred_multi.shape == (5,)

    @pytest.mark.parametrize("model_func,expected_type", [
        (train_linear_regression, LinearRegression),
        (train_random_forest, RandomForestRegressor)
    ])
    def test_model_types(self, sample_data, model_func, expected_type):
        """Test that correct model types are returned."""
        X, y = sample_data
        model = model_func(X, y)
        assert isinstance(model, expected_type)

    def test_random_forest_feature_importance(self, sample_data):
        """Test that Random Forest provides feature importance."""
        X, y = sample_data
        model = train_random_forest(X, y)

        importance = model.feature_importances_
        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)  # All importances should be non-negative
        assert abs(importance.sum() - 1.0) < 1e-6  # Should sum to 1

    def test_model_evaluation_metrics_calculation(self, sample_data):
        """Test that evaluation metrics are calculated correctly."""
        X, y = sample_data
        model = train_linear_regression(X, y)

        # Manual calculation
        y_pred = model.predict(X)
        manual_mse = mean_squared_error(y, y_pred)
        manual_r2 = r2_score(y, y_pred)

        # Function calculation
        metrics = evaluate_model(model, X, y)

        assert abs(metrics['MSE'] - manual_mse) < 1e-10
        assert abs(metrics['R2'] - manual_r2) < 1e-10
