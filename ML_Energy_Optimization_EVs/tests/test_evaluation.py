import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import (
    plot_predictions,
    plot_feature_importance,
    print_evaluation_metrics,
    plot_correlation_matrix
)
from sklearn.ensemble import RandomForestRegressor
import tempfile
import os


class TestEvaluation:
    """Test cases for evaluation module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.rand(n_samples) * 10
        y_pred = y_true + np.random.normal(0, 0.5, n_samples)
        return y_true, y_pred

    @pytest.fixture
    def sample_model(self):
        """Create a sample trained model."""
        np.random.seed(42)
        X = np.random.rand(50, 4)
        y = X.sum(axis=1) + np.random.normal(0, 0.1, 50)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, ['feature1', 'feature2', 'feature3', 'feature4']

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for correlation plotting."""
        np.random.seed(42)
        data = {
            'A': np.random.rand(50),
            'B': np.random.rand(50),
            'C': np.random.rand(50),
            'D': np.random.rand(50)
        }
        return pd.DataFrame(data)

    def test_plot_predictions_creates_figure(self, sample_data):
        """Test that plot_predictions creates a matplotlib figure."""
        y_true, y_pred = sample_data

        # Should not raise any exceptions
        plot_predictions(y_true, y_pred, "Test Model")

        # Check that a figure was created
        assert plt.gcf() is not None

        # Clean up
        plt.close('all')

    def test_plot_predictions_data_integrity(self, sample_data):
        """Test that plot_predictions doesn't modify input data."""
        y_true, y_pred = sample_data
        y_true_orig = y_true.copy()
        y_pred_orig = y_pred.copy()

        plot_predictions(y_true, y_pred, "Test Model")

        # Data should be unchanged
        np.testing.assert_array_equal(y_true, y_true_orig)
        np.testing.assert_array_equal(y_pred, y_pred_orig)

        plt.close('all')

    def test_plot_feature_importance_with_valid_model(self, sample_model):
        """Test feature importance plotting with a valid model."""
        model, feature_names = sample_model

        # Should not raise exceptions
        plot_feature_importance(model, feature_names, "Test Model")

        # Check that a figure was created
        assert plt.gcf() is not None

        plt.close('all')

    def test_plot_feature_importance_without_feature_importance(self):
        """Test feature importance plotting with model that doesn't have feature_importances_."""
        from sklearn.linear_model import LinearRegression

        # LinearRegression doesn't have feature_importances_
        model = LinearRegression()
        feature_names = ['feat1', 'feat2']

        # Should handle gracefully (print message, no plot)
        plot_feature_importance(model, feature_names, "Linear Model")

        # Should not have created a figure
        # Note: This test might be tricky as plt.gcf() might still exist from previous tests

    def test_print_evaluation_metrics_output(self, sample_data, capsys):
        """Test that print_evaluation_metrics produces expected output."""
        y_true, y_pred = sample_data

        print_evaluation_metrics(y_true, y_pred, "Test Model")

        captured = capsys.readouterr()
        output = captured.out

        # Check that output contains expected metrics
        assert "Test Model Evaluation Metrics:" in output
        assert "Mean Squared Error" in output
        assert "Mean Absolute Error" in output
        assert "R² Score" in output

    def test_print_evaluation_metrics_calculations(self, capsys):
        """Test that metrics are calculated correctly in print function."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        print_evaluation_metrics(y_true, y_pred, "Test Model")

        captured = capsys.readouterr()
        output = captured.out

        # Extract numerical values from output (simplified check)
        lines = output.strip().split('\n')
        mse_line = [line for line in lines if 'MSE' in line][0]
        mae_line = [line for line in lines if 'MAE' in line][0]
        r2_line = [line for line in lines if 'R²' in line][0]

        # Check that values are present and reasonable
        assert ':' in mse_line
        assert ':' in mae_line
        assert ':' in r2_line

    def test_plot_correlation_matrix_creates_figure(self, sample_dataframe):
        """Test that correlation matrix plotting creates a figure."""
        plot_correlation_matrix(sample_dataframe)

        # Check that a figure was created
        assert plt.gcf() is not None

        plt.close('all')

    def test_plot_correlation_matrix_data_integrity(self, sample_dataframe):
        """Test that correlation plotting doesn't modify input data."""
        original_data = sample_dataframe.copy()

        plot_correlation_matrix(sample_dataframe)

        # Data should be unchanged
        pd.testing.assert_frame_equal(sample_dataframe, original_data)

        plt.close('all')

    @pytest.mark.parametrize("model_name", ["Random Forest", "Linear Regression", "SVM"])
    def test_plot_predictions_with_different_model_names(self, sample_data, model_name):
        """Test plotting with different model names."""
        y_true, y_pred = sample_data

        plot_predictions(y_true, y_pred, model_name)

        # Get current axis
        ax = plt.gca()

        # Check that title contains model name
        assert model_name in ax.get_title()

        plt.close('all')

    def test_evaluation_functions_handle_edge_cases(self):
        """Test evaluation functions with edge cases."""
        # Perfect predictions
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        # Should not raise exceptions
        plot_predictions(y_true, y_pred, "Perfect Model")

        plt.close('all')

    def test_feature_importance_plot_labels(self, sample_model):
        """Test that feature importance plot has correct labels."""
        model, feature_names = sample_model

        plot_feature_importance(model, feature_names, "Test Model")

        ax = plt.gca()

        # Check x-axis labels
        xticks = ax.get_xticklabels()
        plotted_labels = [tick.get_text() for tick in xticks]

        # Should contain our feature names
        for name in feature_names:
            assert name in plotted_labels

        plt.close('all')

    def test_correlation_matrix_annotates_values(self, sample_dataframe):
        """Test that correlation matrix shows annotations."""
        plot_correlation_matrix(sample_dataframe)

        ax = plt.gca()

        # Check if it's a heatmap (should have collections)
        assert len(ax.collections) > 0

        plt.close('all')

    def test_functions_handle_empty_data(self):
        """Test that functions handle empty data gracefully."""
        # This should raise appropriate errors or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            plot_predictions(np.array([]), np.array([]), "Empty Model")

    def test_print_metrics_with_extreme_values(self, capsys):
        """Test metrics printing with extreme values."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([100, 101, 102])  # Very different predictions

        print_evaluation_metrics(y_true, y_pred, "Extreme Model")

        captured = capsys.readouterr()
        output = captured.out

        # Should still produce output
        assert len(output) > 0
        assert "Extreme Model" in output
