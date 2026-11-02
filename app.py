"""
EV Battery Health Prediction - Streamlit Web Application
M.Tech (AIML) - Machine Learning Assignment 2

Student: PRAKASH B
Student ID: 2026AJ05668

This app allows users to:
1. Upload test data (CSV)
2. Select a classification model
3. View evaluation metrics
4. Display confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="EV Battery Health Prediction",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔋 EV Battery Health Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Class Classification Model Comparison</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Model Configuration")
st.sidebar.markdown("---")

# Model information
st.sidebar.info("""
**Student:** PRAKASH B
**ID:** 2026AJ05668
**Course:** M.Tech (AIML) - ML
**Assignment:** 2
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Instructions")
st.sidebar.markdown("""
1. Upload the test data CSV file
2. Select a classification model
3. View evaluation metrics
4. Analyze confusion matrix
5. Review classification report
""")

# Model names and file paths
MODEL_INFO = {
    "Logistic Regression": {
        "file": "model/logistic_regression.pkl",
        "scaled": True,
        "description": "Linear model with L2 regularization. Best overall performer (Accuracy: 94.93%, AUC: 98.86%)."
    },
    "Decision Tree": {
        "file": "model/decision_tree.pkl",
        "scaled": False,
        "description": "Tree-based model with interpretable decision rules (Accuracy: 94.58%, AUC: 95.60%)."
    },
    "K-Nearest Neighbors": {
        "file": "model/knn.pkl",
        "scaled": True,
        "description": "Instance-based learning with k=5 neighbors (Accuracy: 90.42%, AUC: 92.97%)."
    },
    "Naive Bayes": {
        "file": "model/naive_bayes.pkl",
        "scaled": True,
        "description": "Gaussian Naive Bayes for continuous features (Accuracy: 89.13%, AUC: 97.67%)."
    },
    "Support Vector Machine": {
        "file": "model/svm.pkl",
        "scaled": True,
        "description": "Linear SVM with maximum margin classifier (Accuracy: 93.65%, AUC: 98.31%)."
    }
}

# Load preprocessing artifacts
@st.cache_resource
def load_artifacts():
    """Load scaler and label encoder"""
    try:
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('target_label_encoder.pkl')
        return scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

# Load model
@st.cache_resource
def load_model(model_path):
    """Load trained model from file"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_scores, model_classes):
    """Calculate all required evaluation metrics"""
    metrics = {}

    # Accuracy
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)

    # AUC Score
    try:
        if len(model_classes) == 2:
            # Binary classification
            if hasattr(y_scores, 'shape') and len(y_scores.shape) > 1:
                auc = roc_auc_score(y_true, y_scores[:, 1])
            else:
                auc = roc_auc_score(y_true, y_scores)
        else:
            # Multi-class classification
            if hasattr(y_scores, 'shape') and len(y_scores.shape) > 1 and y_scores.shape[1] == len(model_classes):
                # predict_proba output
                auc = roc_auc_score(y_true, y_scores, multi_class='ovr', average='weighted')
            else:
                # decision_function output
                y_true_binary = label_binarize(y_true, classes=model_classes)
                auc = roc_auc_score(y_true_binary, y_scores, average='weighted')
        metrics['AUC'] = auc
    except Exception as e:
        metrics['AUC'] = np.nan
        st.warning(f"AUC calculation warning: {e}")

    # Precision, Recall, F1 (weighted for multi-class)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Matthews Correlation Coefficient
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    return metrics

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    return fig

# Main app
def main():
    # Load artifacts
    scaler, label_encoder = load_artifacts()

    if scaler is None or label_encoder is None:
        st.error("⚠️ Required preprocessing artifacts not found. Please ensure 'scaler.pkl' and 'target_label_encoder.pkl' are in the directory.")
        return

    # Get class names
    class_names = label_encoder.classes_

    # File uploader
    st.markdown("## 📁 Step 1: Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload your test data CSV file (must contain features and target column)",
        type=['csv'],
        help="Upload the test_data.csv file included in the repository"
    )

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

            # Show data preview
            with st.expander("📊 View Data Preview (First 10 rows)"):
                st.dataframe(df.head(10))

            # Check for target column
            target_column = 'failure_risk_label'
            if target_column not in df.columns:
                st.error(f"⚠️ Target column '{target_column}' not found in uploaded file.")
                return

            # Separate features and target
            X_test = df.drop(columns=[target_column])
            y_test = df[target_column]

            st.info(f"📊 Features: {X_test.shape[1]} | Samples: {X_test.shape[0]} | Classes: {y_test.nunique()}")

            # Model selection
            st.markdown("## 🤖 Step 2: Select Classification Model")
            selected_model_name = st.selectbox(
                "Choose a model to evaluate:",
                list(MODEL_INFO.keys()),
                help="Select from 5 trained classification models"
            )

            # Display model description
            st.info(f"ℹ️ {MODEL_INFO[selected_model_name]['description']}")

            # Load selected model
            model_path = MODEL_INFO[selected_model_name]['file']
            needs_scaling = MODEL_INFO[selected_model_name]['scaled']

            model = load_model(model_path)

            if model is None:
                st.error(f"⚠️ Model file not found: {model_path}")
                return

            # Prepare features
            if needs_scaling:
                X_test_prepared = scaler.transform(X_test)
            else:
                X_test_prepared = X_test.values

            # Make predictions
            with st.spinner('🔄 Making predictions...'):
                y_pred = model.predict(X_test_prepared)

                # Get prediction scores
                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(X_test_prepared)
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test_prepared)
                else:
                    y_scores = None

            st.success("✅ Predictions completed!")

            # Calculate metrics
            st.markdown("## 📈 Step 3: Evaluation Metrics")

            if y_scores is not None:
                metrics = calculate_metrics(y_test, y_pred, y_scores, model.classes_)
            else:
                # Calculate metrics without AUC
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'AUC': np.nan,
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'MCC': matthews_corrcoef(y_test, y_pred)
                }

            # Display metrics in columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🎯 Accuracy", f"{metrics['Accuracy']:.4f}")
                st.metric("📊 Precision", f"{metrics['Precision']:.4f}")

            with col2:
                if not np.isnan(metrics['AUC']):
                    st.metric("📈 AUC Score", f"{metrics['AUC']:.4f}")
                else:
                    st.metric("📈 AUC Score", "N/A")
                st.metric("🔍 Recall", f"{metrics['Recall']:.4f}")

            with col3:
                st.metric("⚖️ F1 Score", f"{metrics['F1 Score']:.4f}")
                st.metric("🔗 MCC", f"{metrics['MCC']:.4f}")

            # Detailed metrics table
            with st.expander("📋 View Detailed Metrics Table"):
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df.style.format("{:.4f}"))

            # Confusion Matrix
            st.markdown("## 🔲 Step 4: Confusion Matrix")

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = plot_confusion_matrix(y_test, y_pred, class_names)
                st.pyplot(fig)

            with col2:
                st.markdown("### 📊 Matrix Interpretation")
                st.markdown("""
                The confusion matrix shows:
                - **Diagonal**: Correct predictions
                - **Off-diagonal**: Misclassifications

                **Rows**: True labels
                **Columns**: Predicted labels
                """)

                # Calculate total accuracy from confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                total_correct = np.trace(cm)
                total_samples = np.sum(cm)
                st.metric("Total Correct", f"{total_correct}/{total_samples}")

            # Classification Report
            st.markdown("## 📄 Step 5: Classification Report")

            report = classification_report(y_test, y_pred,
                                         target_names=class_names,
                                         output_dict=True,
                                         zero_division=0)

            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', subset=['f1-score']))

            # Per-class analysis
            with st.expander("📊 Per-Class Performance Analysis"):
                for class_name in class_names:
                    if class_name in report:
                        st.markdown(f"### Class: {class_name}")
                        class_metrics = report[class_name]

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Precision", f"{class_metrics['precision']:.4f}")
                        col2.metric("Recall", f"{class_metrics['recall']:.4f}")
                        col3.metric("F1-Score", f"{class_metrics['f1-score']:.4f}")
                        col4.metric("Support", f"{int(class_metrics['support'])}")

                        st.markdown("---")

            # Download predictions
            st.markdown("## 💾 Download Predictions")

            predictions_df = pd.DataFrame({
                'True_Label': y_test.values,
                'Predicted_Label': y_pred,
                'True_Class': [class_names[i] for i in y_test.values],
                'Predicted_Class': [class_names[i] for i in y_pred]
            })

            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.exception(e)

    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a test data CSV file to begin evaluation.")

        st.markdown("### 📝 Expected File Format")
        st.markdown("""
        The test data CSV should contain:
        - All 51 feature columns (after preprocessing)
        - One target column: `failure_risk_label`
        - No missing values

        You can use the `test_data.csv` file included in the repository.
        """)

        # Show model comparison
        st.markdown("### 🏆 Model Performance Comparison")

        comparison_data = {
            'Model': list(MODEL_INFO.keys()),
            'Accuracy': [0.9493, 0.9458, 0.9042, 0.8913, 0.9365],
            'AUC': [0.9886, 0.9560, 0.9297, 0.9767, 0.9831],
            'Precision': [0.9486, 0.9456, 0.8844, 0.9350, 0.9332],
            'Recall': [0.9493, 0.9458, 0.9042, 0.8913, 0.9365],
            'F1': [0.9488, 0.9457, 0.8886, 0.9054, 0.9261],
            'MCC': [0.7999, 0.7884, 0.5729, 0.6826, 0.7363]
        }

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>EV Battery Health Prediction System</strong></p>
    <p>M.Tech (AIML) - Machine Learning Assignment 2</p>
    <p>Student: PRAKASH B | ID: 2026AJ05668</p>
    <p>BITS Pilani Work Integrated Learning Programmes</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
