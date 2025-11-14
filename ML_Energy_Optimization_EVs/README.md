# Machine Learning for Energy Consumption Reduction in Hybrid Energy Storage Electric Vehicles

This project implements a machine learning approach to predict and optimize energy consumption in hybrid energy storage electric vehicles (HESS EVs). The system uses a combination of battery and supercapacitor storage to minimize energy loss and improve efficiency.

## Project Structure

```
ML_Energy_Optimization_EVs/
├── data_preprocessing.py    # Data generation and preprocessing module
├── model_training.py        # ML model training and evaluation functions
├── evaluation.py            # Visualization and detailed evaluation functions
├── main.py                  # Main script to run the complete pipeline
├── notebook.ipynb           # Jupyter notebook for interactive demonstration
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Features

- **Synthetic Data Generation**: Creates realistic vehicle data including speed, acceleration, load, and state of charge for battery and supercapacitor.
- **Machine Learning Models**: Implements Random Forest Regressor for energy consumption prediction.
- **Modular Design**: Clean, organized code with separate modules for different functionalities.
- **Comprehensive Evaluation**: Includes metrics like MSE, MAE, R² score, and visualizations.
- **Feature Importance Analysis**: Identifies which factors most influence energy consumption.

## Installation

1. Clone or download the project directory.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Script

Execute the main script to run the complete ML pipeline:

```bash
python main.py
```

This will:
- Generate synthetic data
- Preprocess the data
- Train a Random Forest model
- Evaluate the model performance
- Generate visualizations
- Save the trained model

### Using the Jupyter Notebook

Open `notebook.ipynb` in Jupyter Lab/Notebook for an interactive experience:

```bash
jupyter notebook notebook.ipynb
```

The notebook provides step-by-step execution with explanations and visualizations.

### Running the Web Application

To launch the interactive web interface:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

This will start a local web server. Open the provided URL in your browser to access the energy consumption predictor interface.

**Web App Features:**
- Input vehicle parameters using sliders
- Real-time energy consumption prediction
- Feature importance visualization
- Model performance information

## Key Components

### Data Preprocessing (`data_preprocessing.py`)
- Generates synthetic vehicle data
- Handles data splitting and feature scaling

### Model Training (`model_training.py`)
- Trains Random Forest and Linear Regression models
- Provides model evaluation and saving/loading functionality

### Evaluation (`evaluation.py`)
- Plots actual vs predicted values
- Displays feature importance
- Generates correlation matrices
- Prints detailed performance metrics

## Model Performance

The Random Forest model typically achieves:
- R² Score: ~0.85-0.90
- MSE: Low values indicating good fit
- Feature importance highlighting speed and load as key factors

## Future Enhancements

- Integrate real vehicle data
- Implement more advanced ML models (e.g., Neural Networks)
- Add real-time prediction capabilities
- Optimize for specific vehicle types
- Include weather and terrain data

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning algorithms
- matplotlib: Plotting
- seaborn: Statistical visualizations
- jupyter: Interactive notebooks

## License

This project is for educational purposes. Feel free to modify and use as needed.
