from data_preprocessing import generate_synthetic_data, preprocess_data
from model_training import train_random_forest, evaluate_model, save_model
from evaluation import plot_predictions, plot_feature_importance, print_evaluation_metrics, plot_correlation_matrix
from hyperparameter_tuning import tune_random_forest, tune_multiple_models
from model_comparison import ModelComparator
from logging_config import setup_logging, get_logger, log_function_call
import pandas as pd
import sys

# Set up logging
logger = setup_logging()

@log_function_call
def main():
    """
    Main function to run the complete ML pipeline for energy consumption prediction.
    """
    try:
        logger.info("Starting ML pipeline for energy consumption prediction")

        # Generate synthetic data
        logger.info("Generating synthetic data...")
        data = generate_synthetic_data(num_samples=1000)
        logger.info(f"Generated {len(data)} samples with {len(data.columns)} features")

        # Preprocess data
        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")

        # Hyperparameter tuning
        logger.info("Performing hyperparameter tuning...")
        tuning_results = tune_random_forest(X_train, y_train, method='grid', cv=3)
        logger.info(f"Best parameters found: {tuning_results['best_params']}")

        # Train best model
        logger.info("Training optimized Random Forest model...")
        model = tuning_results['best_model']

        # Evaluate model
        logger.info("Evaluating model performance...")
        metrics = evaluate_model(model, X_test, y_test)
        logger.info(f"Model Performance - MSE: {metrics['MSE']:.4f}, R²: {metrics['R2']:.4f}")

        # Get predictions for plotting
        y_pred = model.predict(X_test)

        # Generate plots
        logger.info("Generating evaluation plots...")
        try:
            plot_correlation_matrix(data)
            plot_predictions(y_test, y_pred, "Optimized Random Forest")
            feature_names = data.drop('energy_consumption', axis=1).columns.tolist()
            plot_feature_importance(model, feature_names, "Optimized Random Forest")
        except Exception as e:
            logger.warning(f"Could not generate plots: {str(e)}")

        # Print detailed metrics
        print_evaluation_metrics(y_test, y_pred, "Optimized Random Forest")

        # Model comparison (optional - can be time-consuming)
        try:
            logger.info("Performing model comparison...")
            comparator = ModelComparator()
            comparator.add_models()
            comparator.evaluate_models(X_train, y_train, X_test, y_test, cv=3)

            # Generate comparison report
            report = comparator.generate_comparison_report()
            print("\n" + "="*50)
            print("MODEL COMPARISON REPORT")
            print("="*50)
            print(report)

            # Plot comparison
            comparator.plot_model_comparison('test_mse')

        except Exception as e:
            logger.warning(f"Model comparison failed: {str(e)}")

        # Save model and scaler
        logger.info("Saving trained model...")
        save_model(model, 'optimized_random_forest_model.pkl')
        logger.info("Model saved as 'optimized_random_forest_model.pkl'")

        # Save scaler
        logger.info("Saving scaler...")
        joblib.dump(scaler, 'scaler.pkl')
        logger.info("Scaler saved as 'scaler.pkl'")

        logger.info("ML pipeline completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Pipeline completed successfully!")
        print("📊 Check the generated plots and logs for detailed results")
        print("💾 Model saved as 'optimized_random_forest_model.pkl'")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)
