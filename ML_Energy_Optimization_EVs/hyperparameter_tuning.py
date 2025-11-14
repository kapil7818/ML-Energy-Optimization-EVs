"""
Hyperparameter tuning module for ML models.
Provides functions to optimize model parameters using grid search and cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tune_random_forest(X_train, y_train, method='grid', cv=5, n_iter=20):
    """
    Tune Random Forest hyperparameters using Grid Search or Random Search.

    Parameters:
    X_train (array-like): Training features
    y_train (array-like): Training target
    method (str): 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    cv (int): Number of cross-validation folds
    n_iter (int): Number of iterations for random search

    Returns:
    dict: Best parameters and model performance
    """
    try:
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        # Initialize base model
        rf = RandomForestRegressor(random_state=42)

        if method == 'grid':
            logger.info("Performing Grid Search for Random Forest...")
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_

        elif method == 'random':
            logger.info("Performing Random Search for Random Forest...")
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            best_score = -random_search.best_score_

        else:
            raise ValueError("Method must be 'grid' or 'random'")

        # Evaluate on training data
        y_pred = best_model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred)
        train_r2 = r2_score(y_train, y_pred)
        train_mae = mean_absolute_error(y_train, y_pred)

        logger.info(f"Best parameters: {best_params}")
        logger.info(".4f")
        logger.info(".4f")
        logger.info(".4f")

        return {
            'best_model': best_model,
            'best_params': best_params,
            'cv_mse': best_score,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'train_mae': train_mae,
            'method': method
        }

    except Exception as e:
        logger.error(f"Error in Random Forest tuning: {str(e)}")
        raise


def tune_multiple_models(X_train, y_train, cv=5):
    """
    Compare multiple ML models with default and tuned parameters.

    Parameters:
    X_train (array-like): Training features
    y_train (array-like): Training target
    cv (int): Number of cross-validation folds

    Returns:
    dict: Model comparison results
    """
    try:
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'SVR': SVR()
        }

        tuned_params = {
            'Ridge Regression': {'alpha': [0.1, 1.0, 10.0]},
            'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1.0]},
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            },
            'SVR': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf']
            }
        }

        results = {}

        for name, model in models.items():
            logger.info(f"Evaluating {name}...")

            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='neg_mean_squared_error'
            )
            cv_mse = -cv_scores.mean()
            cv_std = cv_scores.std()

            # Train model and get predictions
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)

            train_mse = mean_squared_error(y_train, y_pred)
            train_r2 = r2_score(y_train, y_pred)
            train_mae = mean_absolute_error(y_train, y_pred)

            results[name] = {
                'cv_mse_mean': cv_mse,
                'cv_mse_std': cv_std,
                'train_mse': train_mse,
                'train_r2': train_r2,
                'train_mae': train_mae,
                'model': model
            }

            # Hyperparameter tuning for applicable models
            if name in tuned_params:
                logger.info(f"Tuning hyperparameters for {name}...")
                grid_search = GridSearchCV(
                    model, tuned_params[name], cv=cv,
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)

                tuned_model = grid_search.best_estimator_
                tuned_y_pred = tuned_model.predict(X_train)

                results[name]['tuned_model'] = tuned_model
                results[name]['tuned_params'] = grid_search.best_params_
                results[name]['tuned_cv_mse'] = -grid_search.best_score_
                results[name]['tuned_train_mse'] = mean_squared_error(y_train, tuned_y_pred)
                results[name]['tuned_train_r2'] = r2_score(y_train, tuned_y_pred)
                results[name]['tuned_train_mae'] = mean_absolute_error(y_train, tuned_y_pred)

        # Find best model
        best_model_name = min(results.keys(),
                            key=lambda x: results[x]['cv_mse_mean'])
        best_model = results[best_model_name]['model']

        logger.info(f"Best model: {best_model_name} with CV MSE: {results[best_model_name]['cv_mse_mean']:.4f}")

        return {
            'results': results,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'model_names': list(results.keys())
        }

    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        raise


def get_feature_importance_analysis(model, feature_names, X_train, y_train):
    """
    Analyze feature importance for tree-based models.

    Parameters:
    model: Trained model with feature_importances_
    feature_names (list): List of feature names
    X_train (array-like): Training features
    y_train (array-like): Training target

    Returns:
    dict: Feature importance analysis
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not support feature importance analysis")
            return None

        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        feature_analysis = {
            'feature_names': [feature_names[i] for i in indices],
            'importance_values': importance[indices],
            'importance_ranks': list(range(1, len(feature_names) + 1))
        }

        # Calculate cumulative importance
        cumulative_importance = np.cumsum(importance[indices])
        feature_analysis['cumulative_importance'] = cumulative_importance

        # Find number of features for 95% importance
        n_features_95 = np.where(cumulative_importance >= 0.95)[0][0] + 1
        feature_analysis['n_features_95_percent'] = n_features_95

        logger.info(f"Top 5 important features: {feature_analysis['feature_names'][:5]}")
        logger.info(f"Number of features needed for 95% importance: {n_features_95}")

        return feature_analysis

    except Exception as e:
        logger.error(f"Error in feature importance analysis: {str(e)}")
        raise


def perform_cross_validation_analysis(model, X_train, y_train, cv_folds=[3, 5, 10]):
    """
    Perform cross-validation analysis with different fold numbers.

    Parameters:
    model: ML model to evaluate
    X_train (array-like): Training features
    y_train (array-like): Training target
    cv_folds (list): List of CV fold numbers to test

    Returns:
    dict: Cross-validation analysis results
    """
    try:
        cv_results = {}

        for folds in cv_folds:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=folds, scoring='neg_mean_squared_error'
            )
            cv_results[f'{folds}_fold'] = {
                'mean_mse': -scores.mean(),
                'std_mse': scores.std(),
                'scores': -scores
            }
            logger.info(f"{folds}-fold CV - Mean MSE: {-scores.mean():.4f} (+/- {scores.std()*2:.4f})")

        return cv_results

    except Exception as e:
        logger.error(f"Error in cross-validation analysis: {str(e)}")
        raise
