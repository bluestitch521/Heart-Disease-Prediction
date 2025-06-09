# Heart-Disease-Prediction

This repository contains a complete pipeline for predicting heart disease using machine learning models, with a focus on model explainability and feature selection. The project includes data preprocessing, variable analysis, model training, cross-validation, feature reduction (via RFE), and explainability analysis using SHAP values.

##  Repository Structure
- Cross Validation/ # Scripts and results for 5-fold cross-validation on all models
- Prediction Models/ # Implementation of six ML models: LR, KNN, SVM, RF, XGBoost, MLP
- RFE+SHAP/ # Recursive Feature Elimination and SHAP-based feature importance analysis
- Variable Analysis/ # Univariate, bivariate, and multivariate exploration of dataset variables
- README.md # Project overview and usage instructions


##  Project Objective

The aim of this project is to predict the presence of heart disease using a publicly available dataset (UCI Cleveland Heart Disease Dataset) and to:
- Evaluate and compare multiple classification models
- Select the most informative features using RFE
- Interpret model predictions using SHAP values
- Assess model performance through cross-validation

##  Models Implemented

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)

##  Key Analyses

- **Variable Analysis**: Statistical exploration of individual and combined features to understand data characteristics
- **Cross Validation**: 5-fold stratified CV for robust model performance estimation
- **RFE**: Recursive Feature Elimination for dimensionality reduction and performance enhancement
- **SHAP**: Global and local explainability through Shapley values for model transparency

##  Performance Metrics

Each model was evaluated using:
- Accuracy
- Macro F1-Score
- ROC-AUC
- Standard deviation across CV folds

##  Results Summary

| Model             | Accuracy | Macro-F1 | ROC-AUC |
|------------------|----------|----------|---------|
| Logistic Regression | 0.87   | 0.86     | 0.88    |
| XGBoost            | **0.90**| **0.89** | **0.94**|
| Random Forest      | 0.88    | 0.87     | 0.92    |
| MLP                | 0.87    | 0.85     | 0.91    |

##  Dependencies

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- shap

You can install the required packages using:

```bash
pip install -r requirements.txt



