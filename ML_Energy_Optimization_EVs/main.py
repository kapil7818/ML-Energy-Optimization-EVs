from data_preprocessing import generate_synthetic_data, preprocess_data
from model_training import train_random_forest, evaluate_model, save_model
from evaluation import plot_predictions, plot_feature_importance, print_evaluation_metrics, plot_correlation_matrix
import pandas as pd

def main():
    """
    Main function to run the ML pipeline for energy consumption prediction.
    """
    print("Generating synthetic data...")
    data = generate_synthetic_data(num_samples=1000)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    print("Training Random Forest model...")
    model = train_random_forest(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Model Performance: {metrics}")

    # Get predictions for plotting
    y_pred = model.predict(X_test)

    print("Generating plots...")
    # Plot correlation matrix
    plot_correlation_matrix(data)

    # Plot predictions
    plot_predictions(y_test, y_pred, "Random Forest")

    # Plot feature importance
    feature_names = data.drop('energy_consumption', axis=1).columns.tolist()
    plot_feature_importance(model, feature_names, "Random Forest")

    # Print detailed metrics
    print_evaluation_metrics(y_test, y_pred, "Random Forest")

    print("Saving model...")
    save_model(model, 'random_forest_model.pkl')

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
