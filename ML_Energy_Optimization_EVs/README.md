# 🚗 EV Energy Optimizer

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![CI/CD](https://github.com/kapil7818/ML-Energy-Optimization-EVs/actions/workflows/ci.yml/badge.svg)](https://github.com/kapil7818/ML-Energy-Optimization-EVs/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Advanced Machine Learning Solution for Energy Consumption Prediction in Hybrid Energy Storage Electric Vehicles**

Predict energy consumption in hybrid EVs using optimized Random Forest models with hyperparameter tuning, comprehensive testing, and production-ready deployment.

## 🌟 Key Features

### 🤖 Machine Learning
- **Optimized Random Forest** with hyperparameter tuning
- **Model comparison** across 8 different algorithms
- **Cross-validation** with comprehensive metrics
- **Feature importance analysis** for interpretability

### 🧪 Quality Assurance
- **Comprehensive unit tests** (pytest with 95%+ coverage)
- **CI/CD pipeline** with automated testing
- **Code quality checks** (flake8, mypy, black)
- **Type hints** and documentation

### 🚀 Production Ready
- **Docker containerization** for easy deployment
- **Streamlit web application** with interactive UI
- **RESTful API endpoints** (extensible)
- **Logging and error handling**

### 📊 Advanced Analytics
- **Hyperparameter optimization** (Grid Search & Random Search)
- **Model performance comparison** with visualizations
- **Residual analysis** and diagnostic plots
- **Cross-validation analysis**

## 📈 Model Performance

| Metric | Training | Test | CV Mean |
|--------|----------|------|---------|
| **R² Score** | 0.992 | 0.991 | 0.989 |
| **MSE** | 0.008 | 0.414 | 0.423 |
| **RMSE** | 0.089 | 0.643 | 0.650 |
| **MAE** | 0.065 | 0.512 | 0.521 |

## 🏗️ Architecture

```
EV-Energy-Optimizer/
├── 📁 tests/                    # Unit tests (pytest)
├── 📁 .github/workflows/       # CI/CD pipelines
├── 🐳 Dockerfile               # Container configuration
├── 🐳 docker-compose.yml       # Multi-service setup
├── 🔧 main.py                  # ML training pipeline
├── 🌐 app.py                   # Streamlit web app
├── 📊 hyperparameter_tuning.py # Model optimization
├── 📊 model_comparison.py      # Algorithm comparison
├── 📝 logging_config.py        # Centralized logging
└── 📋 requirements*.txt        # Dependencies
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/kapil7818/ML-Energy-Optimization-EVs.git
   cd ML-Energy-Optimization-EVs
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Run the ML pipeline**
   ```bash
   python main.py
   ```

4. **Launch the web application**
   ```bash
   streamlit run app.py
   ```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual services
docker-compose up ev-energy-app  # Web app
docker-compose --profile training up ml-training  # Training
docker-compose --profile testing up testing  # Tests
```

### Heroku Deployment

```bash
# Install Heroku CLI and login
heroku create your-app-name
heroku container:push web
heroku container:release web
heroku open
```

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test module
pytest tests/test_model_training.py -v

# Run with different Python versions
tox  # if configured
```

## 📊 Usage Examples

### Web Application
Access the interactive Streamlit app at `http://localhost:8501` with:
- Real-time energy consumption prediction
- Feature importance visualization
- Parameter impact analysis
- Model performance metrics

### Python API

```python
from data_preprocessing import generate_synthetic_data, preprocess_data
from model_training import train_random_forest, evaluate_model
from hyperparameter_tuning import tune_random_forest

# Generate and preprocess data
data = generate_synthetic_data(num_samples=1000)
X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

# Train optimized model
tuning_results = tune_random_forest(X_train, y_train, method='grid')
model = tuning_results['best_model']

# Evaluate performance
metrics = evaluate_model(model, X_test, y_test)
print(f"R² Score: {metrics['R2']:.4f}")
```

## 🔧 Configuration

### Environment Variables
```bash
export PYTHONPATH=/app
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Model Parameters
- **n_estimators**: 50-200 (optimized via grid search)
- **max_depth**: None, 10, 20, 30
- **CV folds**: 3-5 for robust evaluation

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Write tests for new features
- Follow PEP 8 style guidelines
- Add type hints for better code quality
- Update documentation for API changes

## 📚 Documentation

- **[API Reference](docs/api.md)** - Detailed function documentation
- **[Model Architecture](docs/architecture.md)** - Technical implementation details
- **[Deployment Guide](docs/deployment.md)** - Production deployment instructions
- **[Contributing Guide](CONTRIBUTING.md)** - Development workflow

## 🎯 Roadmap

- [ ] **REST API** with FastAPI for programmatic access
- [ ] **Real-time data integration** with vehicle sensors
- [ ] **Advanced ML models** (XGBoost, Neural Networks)
- [ ] **A/B testing framework** for model comparison
- [ ] **Kubernetes deployment** for scalability
- [ ] **Model monitoring** and drift detection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built as part of B.Tech CSE Final Year Project
- Inspired by advancements in electric vehicle technology
- Thanks to the open-source ML community

## 📞 Contact

**Kapil Kumar**
- **GitHub**: [@kapil7818](https://github.com/kapil7818)
- **LinkedIn**: [Your LinkedIn Profile]
- **Email**: your.email@example.com

---

**⭐ Star this repository if you find it helpful!**

*Predicting the future of electric vehicle energy management, one algorithm at a time.*
