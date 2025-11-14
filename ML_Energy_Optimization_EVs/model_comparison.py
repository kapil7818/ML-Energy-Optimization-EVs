"""
Model comparison module for evaluating multiple ML algorithms.
Provides comprehensive comparison of different models with visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score, median_absolute_error
)
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Class for comparing multiple ML models."""

    def __init__(self, random_state=42):
        """Initialize the model comparator."""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = float('inf')

    def add_models(self):
        """Add various ML models to compare."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=self.random_state),
            'Lasso Regression': Lasso(random_state=self.random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                random_state=self.random_state
            ),
            'SVR': SVR(kernel='rbf'),
            'KNN': KNeighborsRegressor()
        }
        logger.info(f"Added {len(self.models)} models for comparison")

    def evaluate_models(self, X_train, y_train, X_test, y_test, cv=5):
        """
        Evaluate all models using cross-validation and test set.

        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        cv: Number of CV folds
        """
        try:
            self.results = {}

            for name, model in self.models.items():
                logger.info(f"Evaluating {name}...")

                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
                )

                # Train model on full training set
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate metrics
                metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, cv_scores)

                self.results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'cv_scores': cv_scores,
                    'predictions': {
                        'train': y_train_pred,
                        'test': y_test_pred
                    }
                }

                # Track best model
                if metrics['test_mse'] < self.best_score:
                    self.best_score = metrics['test_mse']
                    self.best_model = name

                logger.info(f"{name} - Test MSE: {metrics['test_mse']:.4f}, R²: {metrics['test_r2']:.4f}")

            logger.info(f"Best model: {self.best_model} with Test MSE: {self.best_score:.4f}")

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def _calculate_metrics(self, y_train, y_train_pred, y_test, y_test_pred, cv_scores):
        """Calculate comprehensive metrics for model evaluation."""
        return {
            # Training metrics
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'train_explained_var': explained_variance_score(y_train, y_train_pred),
            'train_median_ae': median_absolute_error(y_train, y_train_pred),

            # Test metrics
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_explained_var': explained_variance_score(y_test, y_test_pred),
            'test_median_ae': median_absolute_error(y_test, y_test_pred),

            # Cross-validation metrics
            'cv_mse_mean': -cv_scores.mean(),
            'cv_mse_std': cv_scores.std(),
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std())
        }

    def plot_model_comparison(self, metric='test_mse', figsize=(12, 8)):
        """
        Plot comparison of models based on specified metric.

        Parameters:
        metric: Metric to compare ('test_mse', 'test_r2', 'cv_mse_mean', etc.)
        figsize: Figure size for the plot
        """
        try:
            model_names = list(self.results.keys())
            values = [self.results[name]['metrics'][metric] for name in model_names]

            plt.figure(figsize=figsize)

            # Create bar plot
            bars = plt.bar(model_names, values, color='skyblue', alpha=0.8)

            # Highlight best model
            if self.best_model in model_names:
                best_idx = model_names.index(self.best_model)
                bars[best_idx].set_color('darkblue')
                bars[best_idx].set_alpha(1.0)

            plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            plt.xlabel('Models', fontsize=12)
            plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
            plt.xticks(rotation=45, ha='right')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        '.3f', ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            plt.grid(axis='y', alpha=0.3)
            plt.show()

        except Exception as e:
            logger.error(f"Error in plotting model comparison: {str(e)}")
            raise

    def plot_learning_curves(self, model_name, X_train, y_train, figsize=(10, 6)):
        """
        Plot learning curves for a specific model.

        Parameters:
        model_name: Name of the model to plot
        X_train, y_train: Training data
        figsize: Figure size
        """
        try:
            if model_name not in self.results:
                raise ValueError(f"Model {model_name} not found in results")

            model = self.models[model_name]

            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='neg_mean_squared_error'
            )

            train_scores_mean = -train_scores.mean(axis=1)
            train_scores_std = train_scores.std(axis=1)
            val_scores_mean = -val_scores.mean(axis=1)
            val_scores_std = val_scores.std(axis=1)

            plt.figure(figsize=figsize)
            plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training MSE')
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std, alpha=0.1, color='blue')

            plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation MSE')
            plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                           val_scores_mean + val_scores_std, alpha=0.1, color='red')

            plt.title(f'Learning Curves - {model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Training Set Size', fontsize=12)
            plt.ylabel('Mean Squared Error', fontsize=12)
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error in plotting learning curves: {str(e)}")
            raise

    def plot_residuals_analysis(self, model_name, X_test, y_test, figsize=(12, 5)):
        """
        Plot residuals analysis for a specific model.

        Parameters:
        model_name: Name of the model to analyze
        X_test, y_test: Test data
        figsize: Figure size
        """
        try:
            if model_name not in self.results:
                raise ValueError(f"Model {model_name} not found in results")

            model = self.results[model_name]['model']
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred

            fig, axes = plt.subplots(1, 3, figsize=figsize)

            # Residuals vs Predicted
            axes[0].scatter(y_pred, residuals, alpha=0.6, color='blue')
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Predicted')
            axes[0].grid(alpha=0.3)

            # Residuals distribution
            axes[1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Residuals Distribution')
            axes[1].grid(alpha=0.3)

            # Q-Q plot (simplified)
            residuals_sorted = np.sort(residuals)
            theoretical_quantiles = np.random.normal(0, np.std(residuals), len(residuals))
            theoretical_quantiles = np.sort(theoretical_quantiles)

            axes[2].scatter(theoretical_quantiles, residuals_sorted, alpha=0.6, color='purple')
            axes[2].plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
                        [theoretical_quantiles.min(), theoretical_quantiles.max()],
                        'r--', alpha=0.8)
            axes[2].set_xlabel('Theoretical Quantiles')
            axes[2].set_ylabel('Sample Quantiles')
            axes[2].set_title('Q-Q Plot')
            axes[2].grid(alpha=0.3)

            plt.suptitle(f'Residuals Analysis - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error in residuals analysis: {str(e)}")
            raise

    def generate_comparison_report(self):
        """
        Generate a comprehensive comparison report.

        Returns:
        str: Formatted report
        """
        try:
            report = []
            report.append("=" * 60)
            report.append("MACHINE LEARNING MODEL COMPARISON REPORT")
            report.append("=" * 60)
            report.append("")

            # Summary
            report.append("SUMMARY:")
            report.append(f"- Total models compared: {len(self.results)}")
            report.append(f"- Best performing model: {self.best_model}")
            report.append(".4f")
            report.append("")

            # Detailed results
            report.append("DETAILED RESULTS:")
            report.append("-" * 40)

            for name, result in self.results.items():
                metrics = result['metrics']
                report.append(f"\n{name}:")
                report.append(f"  Training Metrics:")
                report.append(".4f")
                report.append(".4f")
                report.append(".4f")
                report.append(f"  Test Metrics:")
                report.append(".4f")
                report.append(".4f")
                report.append(".4f")
                report.append(f"  Cross-Validation:")
                report.append(".4f")
                report.append(".4f")

            report.append("\n" + "=" * 60)

            return "\n".join(report)

        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")
            raise

    def get_best_model_info(self):
        """Get detailed information about the best performing model."""
        if not self.best_model:
            return None

        result = self.results[self.best_model]
        return {
            'name': self.best_model,
            'model': result['model'],
            'metrics': result['metrics'],
            'is_best': True
        }


def quick_model_comparison(X_train, y_train, X_test, y_test, cv=5):
    """
    Quick function to compare models and return the best one.

    Parameters:
    X_train, y_train: Training data
    X_test, y_test: Test data
    cv: Cross-validation folds

    Returns:
    dict: Best model information
    """
    try:
        comparator = ModelComparator()
        comparator.add_models()
        comparator.evaluate_models(X_train, y_train, X_test, y_test, cv)

        return comparator.get_best_model_info()

    except Exception as e:
        logger.error(f"Error in quick model comparison: {str(e)}")
        raise
