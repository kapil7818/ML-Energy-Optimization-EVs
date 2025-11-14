from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training target.

    Returns:
    model: Trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor model.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training target.
    n_estimators (int): Number of trees in the forest.
    random_state (int): Random state for reproducibility.

    Returns:
    model: Trained Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using MSE and R² score.

    Parameters:
    model: Trained model.
    X_test (array-like): Test features.
    y_test (array-like): Test target.

    Returns:
    dict: Evaluation metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'MSE': mse, 'R2': r2}

def save_model(model, filename):
    """
    Save the trained model to a file.

    Parameters:
    model: Trained model.
    filename (str): Path to save the model.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load a trained model from a file.

    Parameters:
    filename (str): Path to the saved model.

    Returns:
    model: Loaded model.
    """
    return joblib.load(filename)
