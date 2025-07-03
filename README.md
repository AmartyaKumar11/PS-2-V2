# Resume Classification ML Pipeline

A comprehensive machine learning pipeline for automated resume classification using both supervised and unsupervised approaches. This project creates labels from semantic similarity scores and evaluates multiple algorithms to classify resumes as fit/unfit for job positions.

## 🚀 Features

- **Automated Resume Parsing**: Extract and clean text from various resume formats
- **Semantic Similarity Analysis**: Uses Sentence-BERT for job-resume matching
- **Multiple ML Approaches**: 6 supervised + 5 unsupervised algorithms
- **Label Creation Strategies**: Binary, percentile-based, and conservative labeling
- **Feature Engineering**: TF-IDF and Count Vectorization with optimization
- **Model Validation**: Cross-validation, correlation analysis, and business logic validation
- **Comprehensive Documentation**: Well-documented Jupyter notebooks with explanations

## 📁 Project Structure

```
PS-2/
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore file
├── LICENSE                            # Project license
│
├── notebooks/                         # Jupyter notebooks
│   ├── resume_preprocessing.ipynb     # Data cleaning and preprocessing
│   └── resume_classification_ml.ipynb # ML pipeline and analysis
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py         # Data preprocessing utilities
│   ├── feature_engineering.py       # Feature extraction functions
│   ├── model_training.py            # ML model training
│   └── evaluation.py                # Model evaluation utilities
│
├── data/                             # Data files
│   ├── raw/                         # Raw resume files
│   ├── processed/                   # Processed datasets
│   │   ├── parsed_resumes.csv
│   │   ├── preprocessed_resumes.csv
│   │   ├── preprocessed_resumes_lemmatized.csv
│   │   └── resume_matching_results_final.csv
│   ├── results/                     # Model outputs
│   │   ├── ml_model_comparison_results.csv
│   │   ├── final_ml_predictions.csv
│   │   └── feature_importance_analysis.csv
│   └── job_descriptions/            # Job description files
│       └── Web-Developer-job-description.txt
│
├── models/                          # Trained models
│   ├── trained_models/
│   │   ├── logistic_regression_binary/
│   │   ├── random_forest_multiclass/
│   │   └── README.md
│   └── model_configs/               # Model configurations
│
├── scripts/                         # Utility scripts
│   ├── parse_resumes_script.py      # Resume parsing script
│   ├── cleanup_project.py           # Project cleanup utility
│   └── run_pipeline.py              # Complete pipeline runner
│
├── docs/                            # Documentation
│   ├── PROJECT_SUMMARY.md           # Detailed project summary
│   ├── API_DOCUMENTATION.md         # API documentation
│   ├── DEPLOYMENT_GUIDE.md          # Deployment instructions
│   └── METHODOLOGY.md               # Technical methodology
│
└── tests/                           # Unit tests
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_feature_engineering.py
    └── test_models.py
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-classification-ml.git
cd resume-classification-ml
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Data Preprocessing
```bash
jupyter notebook notebooks/resume_preprocessing.ipynb
```

### 2. ML Pipeline
```bash
jupyter notebook notebooks/resume_classification_ml.ipynb
```

### 3. Complete Pipeline (Script)
```bash
python scripts/run_pipeline.py --input data/raw --output data/results
```

## 📊 Results Summary

### Model Performance
- **Best Supervised Model**: Logistic Regression (F1: 0.72, Accuracy: 69%)
- **Best Unsupervised Model**: K-Means (Silhouette Score: 0.45)
- **Correlation with Similarity Scores**: 0.68 (Pearson correlation)

### Key Findings
- Technical skills are the strongest predictors
- Experience keywords significantly impact classification
- Conservative labeling improves model reliability
- Ensemble methods show promise for production use

## 🔧 Configuration

### Model Parameters
Models are configured with regularization to prevent overfitting:
- **Logistic Regression**: C=0.1 (strong regularization)
- **Random Forest**: max_depth=10, n_estimators=50
- **XGBoost**: learning_rate=0.1, max_depth=6

### Feature Engineering
- **TF-IDF**: max_features=1000, min_df=2, max_df=0.8
- **N-grams**: (1,2) for unigrams and bigrams
- **Preprocessing**: Lemmatization, stop word removal

## 📈 Usage Examples

### Basic Classification
```python
from src.model_training import ResumeClassifier

classifier = ResumeClassifier()
classifier.load_model('models/best_model.pkl')
prediction = classifier.predict(resume_text)
```

### Batch Processing
```python
from src.evaluation import batch_classify

results = batch_classify(
    resume_files='data/raw/*.pdf',
    job_description='data/job_descriptions/Web-Developer.txt',
    output_file='data/results/predictions.csv'
)
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## 📖 Documentation

- **[Project Summary](docs/PROJECT_SUMMARY.md)**: Comprehensive project overview
- **[Methodology](docs/METHODOLOGY.md)**: Technical approach and algorithms
- **[API Documentation](docs/API_DOCUMENTATION.md)**: Function and class references
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)**: Production deployment instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Amartya Kumar**
- Project developed during PS-2 internship
- Focus: Machine Learning, NLP, Resume Classification

## 🙏 Acknowledgments

- Sentence-BERT for semantic similarity computation
- scikit-learn for machine learning algorithms
- XGBoost for gradient boosting implementation
- Jupyter for interactive development environment

## 📞 Support

For questions and support:
- Create an issue in this repository
- Check the [documentation](docs/)
- Review the [troubleshooting guide](docs/TROUBLESHOOTING.md)

---

**Note**: This project demonstrates a complete ML pipeline with realistic performance expectations and production-ready code structure.
