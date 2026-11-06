# EV Battery Health Prediction - Classification Models Comparison

**M.Tech (AIML) - Machine Learning Assignment 2**
**Student Name:** PRAKASH B
**Student ID:** 2026AJ05668
**Submission Date:** 08-Nov-2026

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [GitHub Repository Link](#github-repository-link)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Model Performance Comparison](#model-performance-comparison)
- [Model Performance Observations](#model-performance-observations)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Technologies Used](#technologies-used)

---

## Problem Statement

Electric vehicle (EV) batteries degrade over time due to various factors including charging patterns, environmental conditions, and usage behavior. Predicting battery failure risk is critical for:
- **Preventive Maintenance**: Identifying high-risk batteries before catastrophic failure
- **Fleet Management**: Optimizing vehicle utilization based on battery health
- **Warranty Claims**: Data-driven assessment of battery degradation patterns
- **Customer Safety**: Early warning systems for battery health issues

This project implements and compares **five machine learning classification models** to predict the **failure risk level** (Low, Medium, High) of EV batteries based on operational telemetry, environmental factors, and battery characteristics. The goal is to identify which model provides the best balance of accuracy, precision, recall, and overall predictive performance for this imbalanced multi-class classification problem.

---

## Dataset Description

**Dataset Name:** EV Battery Health Prediction Dataset
**Source:** Custom synthetic dataset simulating real-world EV battery telemetry
**Classification Type:** Multi-class classification (3 classes: Low, Medium, High failure risk)

### Dataset Characteristics:
- **Total Instances:** 50,000 observations
- **Total Features:** 35 columns (51 features after preprocessing)
- **Feature Count:** 51 features after one-hot encoding
- **Target Variable:** `failure_risk_label` (Low, Medium, High)
- **Class Distribution:**
  - Low Risk: 42,735 instances (85.5%)
  - Medium Risk: 4,409 instances (8.8%)
  - High Risk: 2,856 instances (5.7%)
- **Challenge:** Highly imbalanced dataset requiring careful evaluation

### Feature Categories:

1. **Battery Characteristics (8 features)**
   - Chemistry type (LFP, NMC, NCA, LMO)
   - Manufacturer (8 brands)
   - Vehicle model (8 models)
   - Pack capacity (kWh)
   - BMS firmware version

2. **Temporal Features (4 features)**
   - Battery age (days)
   - Cycle count
   - Manufacture date
   - Service history

3. **Usage Patterns (6 features)**
   - Average daily distance
   - Charging power and frequency
   - Fast charge ratio
   - Depth of discharge
   - State of charge patterns
   - Energy throughput

4. **Environmental Factors (4 features)**
   - Climate zone (cold, temperate, hot_arid, tropical_humid, mixed)
   - Average ambient temperature
   - Humidity percentage
   - Operating altitude

5. **Electrical Measurements (6 features)**
   - Internal resistance (key aging indicator)
   - Average voltage and current
   - Charge time
   - Peak temperature during fast charging
   - Capacity fade percentage

6. **Maintenance & Recalls (3 features)**
   - Service count
   - Last service date
   - Recall flag

### Data Quality:
- **Missing Values:** Present in multiple features (3-37% per feature)
- **Handling:** Median imputation for numerical features, "Unknown" category for categorical features
- **No missing values in target variable**

### Preprocessing Steps:
1. Removed identifier columns and date features to prevent data leakage
2. Imputed missing values (median for numeric, "Unknown" for categorical)
3. One-hot encoded categorical features
4. Standardized features using StandardScaler
5. Stratified train-test split (80-20) to maintain class distribution

---

## GitHub Repository Link

**Repository:** [https://github.com/mewebbie/battery-health-prediction-model](https://github.com/mewebbie/battery-health-prediction-model)

### Repository Contents:
```
battery-health-prediction-model/
│
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── LICENSE                              # MIT License
├── test_data.csv                        # Test dataset for Streamlit app (10,000 samples)
│
├── .streamlit/                          # Streamlit configuration
│   └── config.toml                     # Streamlit app settings
│
├── data/                                # Dataset files
│   ├── ev_battery_health.csv          # Original dataset (50,000 samples)
│   ├── data_dictionary .csv           # Data dictionary
│   └── README_DB.md                   # Database documentation
│
├── models/                              # Trained models and preprocessing
│   ├── logistic_regression.pkl        # Logistic Regression model
│   ├── decision_tree.pkl              # Decision Tree model
│   ├── knn.pkl                        # K-Nearest Neighbors model
│   ├── naive_bayes.pkl                # Naive Bayes model
│   ├── svm.pkl                        # Support Vector Machine model
│   ├── scaler.pkl                     # Feature scaler
│   └── target_label_encoder.pkl       # Label encoder for target
│
├── notebooks/                           # Jupyter notebooks
│   └── app.ipynb                      # Complete notebook with all models
│
└── results/                             # Model evaluation results
    ├── model_comparison.csv           # Model metrics comparison table
    └── model_comparison.png           # Visualization of model comparison
```

---

## Project Structure

### Folder Organization:

**Root Files:**
- **app.py**: Interactive Streamlit web application for model deployment
- **requirements.txt**: All Python package dependencies required for the project
- **README.md**: Comprehensive project documentation
- **LICENSE**: MIT License
- **test_data.csv**: 10,000 test samples (20% of dataset) for Streamlit app evaluation

**.streamlit/**: Streamlit configuration
- **config.toml**: Streamlit app configuration settings

**data/**: Dataset and database documentation
- **ev_battery_health.csv**: Original dataset with 50,000 EV battery observations
- **data_dictionary .csv**: Data dictionary describing all features
- **README_DB.md**: Database schema and usage documentation

**models/**: Trained models and preprocessing artifacts
- **logistic_regression.pkl, decision_tree.pkl, knn.pkl, naive_bayes.pkl, svm.pkl**: Serialized trained models
- **scaler.pkl**: Fitted StandardScaler for feature normalization
- **target_label_encoder.pkl**: Label encoder for target classes

**notebooks/**: Jupyter notebooks for analysis and development
- **app.ipynb**: Complete implementation of data preprocessing, EDA, model training, and evaluation

**results/**: Model evaluation outputs
- **model_comparison.csv**: Comprehensive metrics comparison table
- **model_comparison.png**: Visual comparison of all models

---

## Models Used

Five classification algorithms were implemented and compared:

1. **Logistic Regression**: Linear model with L2 regularization, max_iter=1000
2. **Decision Tree Classifier**: Tree-based model with max_depth=10, min_samples_split=5
3. **K-Nearest Neighbors (KNN)**: Instance-based learning with n_neighbors=5
4. **Naive Bayes**: Gaussian Naive Bayes for continuous features
5. **Support Vector Machine (SVM)**: LinearSVC with dual="auto", max_iter=5000

### Evaluation Metrics:
All models were evaluated using the following six metrics:
1. **Accuracy**: Overall correctness of predictions
2. **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes
3. **Precision**: Proportion of positive predictions that are correct
4. **Recall**: Proportion of actual positives correctly identified
5. **F1 Score**: Harmonic mean of precision and recall
6. **MCC (Matthews Correlation Coefficient)**: Balanced measure for imbalanced datasets

---

## Model Performance Comparison

### Comprehensive Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|-------|
| **Logistic Regression** | **0.9493** | **0.9886** | **0.9486** | **0.9493** | **0.9488** | **0.7999** |
| **Decision Tree** | 0.9458 | 0.9560 | 0.9456 | 0.9458 | 0.9457 | 0.7884 |
| **K-Nearest Neighbors** | 0.9042 | 0.9297 | 0.8844 | 0.9042 | 0.8886 | 0.5729 |
| **Naive Bayes** | 0.8913 | 0.9767 | 0.9350 | 0.8913 | 0.9054 | 0.6826 |
| **Support Vector Machine** | 0.9365 | 0.9831 | 0.9332 | 0.9365 | 0.9261 | 0.7363 |

**Note:** All metrics are calculated using weighted averaging to account for class imbalance.

### Best Performers by Metric:
- **Accuracy**: Logistic Regression (0.9493)
- **AUC**: Logistic Regression (0.9886)
- **Precision**: Logistic Regression (0.9486)
- **Recall**: Logistic Regression (0.9493)
- **F1 Score**: Logistic Regression (0.9488)
- **MCC**: Logistic Regression (0.7999)

---

## Model Performance Observations

### Individual Model Analysis:

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | **Best overall performer** across all six metrics. Achieved the highest accuracy (94.93%), AUC (98.86%), and MCC (0.7999). Demonstrates excellent class separation with balanced precision and recall. Particularly effective at identifying Low-risk batteries while maintaining reasonable performance on Medium and High-risk cases. The model's linear decision boundaries work well with the scaled features, and L2 regularization prevents overfitting. **Recommended for production deployment** due to superior performance, computational efficiency, and interpretability. |
| **Decision Tree** | **Second-best performer** with accuracy of 94.58% and MCC of 0.7884, very close to Logistic Regression. Provides highly interpretable decision rules based on feature thresholds (e.g., internal resistance, cycle count). The tree structure naturally handles non-linear relationships and feature interactions. While AUC (95.60%) is lower than top models, it still delivers strong practical performance. Feature importance analysis reveals that capacity fade, internal resistance, and cycle count are the most critical predictors. No feature scaling required, making it robust to different feature ranges. |
| **K-Nearest Neighbors** | **Lowest performance** among all models with accuracy of 90.42% and MCC of 0.5729. Distance-based approach struggles with the high-dimensional feature space (51 features) and class imbalance. Medium-risk recall is only 26%, indicating the model frequently misclassifies Medium-risk batteries as Low-risk. The uniform weighting of neighbors may not capture complex decision boundaries effectively. Computationally expensive for large datasets (requires storing all training data). Not recommended for this problem despite its simplicity. |
| **Naive Bayes** | Shows **strong AUC (97.67%)** and excellent Medium-risk recall (87%), indicating good class separation. However, the independence assumption is violated in this dataset where features are correlated (e.g., battery age and cycle count). This leads to lower overall accuracy (89.13%) and many false positives for Medium-risk predictions (precision 45%). The model is computationally efficient and performs well when uncertainty quantification is important, but the accuracy trade-off limits its practical deployment. Best suited for real-time screening where high recall for at-risk batteries is critical. |
| **Support Vector Machine** | **Strong overall performance** with accuracy of 93.65% and AUC of 98.31%. LinearSVC finds an optimal hyperplane that maximizes margin between classes. High Medium-risk precision (85%) means fewer false alarms, but lower recall (38%) indicates many Medium-risk cases are missed. The model effectively handles high-dimensional data and is robust to outliers. Training time is acceptable with linear kernel. Good choice when **precision is prioritized over recall**, such as when follow-up diagnostics are expensive. |

### Overall Winner for This Dataset:

**Winner: Logistic Regression**

**Justification:**
- Achieved rank 1.00 across all six metrics (best average rank)
- Highest scores in all six evaluation metrics
- Excellent balance between precision (94.86%) and recall (94.93%)
- Superior AUC (98.86%) indicates excellent class discrimination
- Highest MCC (0.7999) demonstrates robust performance on imbalanced data
- Computationally efficient for both training and inference
- Model coefficients provide interpretability for stakeholders
- Suitable for real-time deployment in production systems

**Key Insights:**
1. The target classes are heavily imbalanced (85.5% Low risk), making MCC and F1 particularly important metrics
2. Logistic Regression's success suggests that linear decision boundaries with properly scaled features are sufficient for this problem
3. Feature engineering and preprocessing (scaling, encoding) were more impactful than complex non-linear models
4. The dataset's high dimensionality (51 features) favors regularized linear models over instance-based methods like KNN
5. Class imbalance handling through stratified sampling was effective across all models

**Recommendations:**
- **Production Deployment**: Use Logistic Regression for its superior performance and efficiency
- **Interpretability Needs**: Use Decision Tree for stakeholder communication and regulatory compliance
- **High-Recall Screening**: Use Naive Bayes for initial filtering when missing a high-risk battery is more costly than false alarms
- **Further Improvement**: Consider ensemble methods (Random Forest, XGBoost) and hyperparameter tuning with cross-validation

---

## Installation and Setup

### Prerequisites:
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/mewebbie/battery-health-prediction-model.git
cd battery-health-prediction-model
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import streamlit; import sklearn; import pandas; print('Installation successful!')"
```

---

## Usage

### Running the Jupyter Notebook:
```bash
jupyter notebook notebooks/app.ipynb
```

### Running the Streamlit App:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

---

## Streamlit App

### Live Deployment:
**Streamlit App Link:** [https://battery-health-prediction-model.streamlit.app/](https://battery-health-prediction-model.streamlit.app/)

### Features:
1. **Dataset Upload**: Upload test_data.csv to evaluate models
2. **Model Selection**: Choose from 5 trained classification models
3. **Evaluation Metrics**: View all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. **Confusion Matrix**: Visual representation of model predictions
5. **Classification Report**: Detailed per-class performance metrics

### How to Use:
1. Navigate to the deployed Streamlit app URL
2. Upload the `test_data.csv` file (included in repository)
3. Select a model from the dropdown menu
4. View comprehensive evaluation results and visualizations
5. Compare different models by switching selections

---

## Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Streamlit**: Web application framework for ML deployment
- **Joblib**: Model serialization and persistence

---

## Model Training Details

### Data Split:
- **Training Set**: 40,000 samples (80%)
- **Testing Set**: 10,000 samples (20%)
- **Split Method**: Stratified sampling to maintain class distribution

### Feature Engineering:
- One-hot encoding for categorical variables (chemistry, manufacturer, vehicle_model, climate_zone, bms_firmware_version)
- Standard scaling for numerical features
- Missing value imputation (median for numeric, "Unknown" for categorical)

### Evaluation Strategy:
- Weighted averaging for multi-class metrics to account for class imbalance
- Confusion matrices for per-class error analysis
- Classification reports for detailed class-level metrics

---

## Key Findings

1. **Logistic Regression emerged as the clear winner**, outperforming all other models across all six evaluation metrics
2. **Class imbalance** (85.5% Low risk) required careful metric selection - MCC and weighted F1 were most informative
3. **Feature scaling** significantly improved model performance, especially for distance-based and linear models
4. **Internal resistance, capacity fade, and cycle count** were identified as the most important predictors
5. **Deployment-ready models** achieved >93% accuracy with robust cross-class performance

---

## Future Improvements

1. **Hyperparameter Tuning**: Grid search or Bayesian optimization for optimal parameters
2. **Ensemble Methods**: Random Forest, Gradient Boosting, or XGBoost for potential performance gains
3. **Feature Engineering**: Create interaction terms and domain-specific derived features
4. **Class Balancing**: SMOTE or class weights to improve minority class prediction
5. **Cross-Validation**: K-fold CV for more robust performance estimation
6. **Temporal Analysis**: Leverage longitudinal data (multiple observations per battery) with time series models
7. **Explainability**: SHAP or LIME for individual prediction explanations

---

## Contact Information

**Student:** PRAKASH B
**Student ID:** 2026AJ05668
**Program:** M.Tech (AIML)
**Institution:** BITS Pilani Work Integrated Learning Programme

---

## License

This project is submitted as part of the Machine Learning course assignment for M.Tech (AIML) program at BITS Pilani.

---

## Acknowledgments

- Dataset inspired by real-world EV battery telemetry patterns
- Assignment designed by BITS Pilani WILP Machine Learning faculty
- Scikit-learn documentation and examples
- Streamlit community for deployment resources

---

**Last Updated:** November 6, 2026
