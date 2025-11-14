import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_predictions(y_test, y_pred, model_name):
    """
    Plot actual vs predicted values.

    Parameters:
    y_test (array-like): Actual values.
    y_pred (array-like): Predicted values.
    model_name (str): Name of the model for the plot title.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Energy Consumption (kWh)')
    plt.ylabel('Predicted Energy Consumption (kWh)')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.grid(True)
    plt.show()

def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models.

    Parameters:
    model: Trained model with feature_importances_ attribute.
    feature_names (list): List of feature names.
    model_name (str): Name of the model for the plot title.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Model {model_name} does not support feature importance plotting.")

def print_evaluation_metrics(y_test, y_pred, model_name):
    """
    Print evaluation metrics.

    Parameters:
    y_test (array-like): Actual values.
    y_pred (array-like): Predicted values.
    model_name (str): Name of the model.
    """
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

def plot_correlation_matrix(data):
    """
    Plot correlation matrix of the dataset.

    Parameters:
    data (pd.DataFrame): Input dataset.
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
