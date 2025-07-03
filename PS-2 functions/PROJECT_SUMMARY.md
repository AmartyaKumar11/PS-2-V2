# Resume Classification ML Pipeline - Project Summary

## Overview
This project implements a comprehensive machine learning pipeline for resume classification using both supervised and unsupervised approaches. The system creates labels from semantic similarity scores and tests multiple algorithms while ensuring realistic performance.

## Project Structure

### Core Notebooks
1. **resume_preprocessing.ipynb** - Data cleaning and preprocessing pipeline
2. **resume_classification_ml.ipynb** - Main ML pipeline with multiple algorithms

### Essential Data Files
- **parsed_resumes.csv** - Raw parsed resumes (input for preprocessing)
- **preprocessed_resumes.csv** - Intermediate processed data
- **preprocessed_resumes_lemmatized.csv** - Lemmatized resumes (used in ML pipeline)
- **resume_matching_results_final.csv** - Final semantic matching results (input for ML)

### Output Files
- **ml_model_comparison_results.csv** - Model comparison results
- **final_ml_predictions.csv** - Final predictions with confidence scores
- **feature_importance_analysis.csv** - Feature importance analysis

### Scripts & Configuration
- **parse_resumes_script.py** - Resume parsing script
- **Web-Developer-job-description.txt** - Job description used in analysis
- **requirements.txt** - Python dependencies
- **trained_models/** - Saved ML models

## Workflow Summary

1. **Data Preprocessing** (resume_preprocessing.ipynb)
   - Parse resumes from various formats
   - Clean and normalize text data
   - Tokenization and lemmatization
   - Create structured datasets

2. **Machine Learning Pipeline** (resume_classification_ml.ipynb)
   - Load preprocessed data and job descriptions
   - Exploratory data analysis of similarity scores
   - Feature engineering with TF-IDF and Count Vectorization
   - Multiple label creation strategies (binary, percentile, conservative)
   - Supervised ML models: Logistic Regression, Random Forest, SVM, XGBoost, Naive Bayes, Gradient Boosting
   - Unsupervised clustering: K-Means, Gaussian Mixture, Agglomerative, DBSCAN
   - Model comparison and validation against similarity scores
   - Feature importance analysis
   - Business validation and deployment recommendations

## Key Results

### Best Performing Models
- **Supervised Learning**: Models show realistic performance (60-75% accuracy)
- **Correlation with Similarity Scores**: Validates model predictions against Sentence-BERT scores
- **Feature Importance**: Technical skills and experience keywords are strong predictors

### Deployment Recommendations
- Use Conservative Labeling for higher quality training data
- Logistic Regression provides good interpretability
- Random Forest offers robust performance
- Consider ensemble methods for production
- Implement regular retraining with new job descriptions

## Technical Stack
- **Python Libraries**: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
- **NLP**: TF-IDF, Count Vectorization, Sentence-BERT (for similarity scores)
- **ML Algorithms**: 6 supervised + 5 unsupervised models
- **Evaluation**: Cross-validation, correlation analysis, business validation

## Next Steps
1. Deploy best-performing models to production
2. Implement A/B testing against current Sentence-BERT approach
3. Set up monitoring for prediction confidence scores
4. Collect feedback to improve labeling strategies
5. Regular model retraining with new data

## Files Cleaned Up
The following unnecessary files have been identified for removal:
- Final.csv (as requested)
- Unused job description files (JD_Data_Analyst.txt, etc.)
- Documentation files (README.md, GITHUB_SETUP files, etc.)
- Build scripts (setup.py, upload_to_github.bat)
- Unused reference files (skills_it.txt)

## Contact & Support
This project provides a robust foundation for resume screening applications with clear documentation and reproducible results.
