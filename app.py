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

    # File uploader at the top (always visible)
    st.markdown("## 📁 Upload Test Data")
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

            # Check for target column
            target_column = 'failure_risk_label'
            if target_column not in df.columns:
                st.error(f"⚠️ Target column '{target_column}' not found in uploaded file.")
                return

            # Separate features and target
            X_test = df.drop(columns=[target_column])
            y_test = df[target_column]

            # Model selection
            st.markdown("## 🤖 Select Classification Model")
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

            # Calculate classification report for later use
            report = classification_report(y_test, y_pred,
                                         target_names=class_names,
                                         output_dict=True,
                                         zero_division=0)

            # Create tabs for better organization
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview",
                "📈 Performance Metrics",
                "🔲 Confusion Matrix",
                "📄 Classification Report",
                "💾 Export Results"
            ])

            # TAB 1: Overview
            with tab1:
                st.markdown("### 🎯 Quick Summary")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📋 Total Samples", f"{X_test.shape[0]:,}")
                    st.metric("🔢 Features", X_test.shape[1])
                with col2:
                    st.metric("🎯 Classes", y_test.nunique())
                    st.metric("🤖 Model", selected_model_name)
                with col3:
                    st.metric("✅ Accuracy", f"{metrics['Accuracy']:.2%}")
                    st.metric("📈 AUC", f"{metrics['AUC']:.4f}" if not np.isnan(metrics['AUC']) else "N/A")

                st.markdown("---")

                # Data preview
                st.markdown("### 📊 Data Preview")
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.dataframe(df.head(10), use_container_width=True)

                with col2:
                    st.markdown("#### 📈 Class Distribution")
                    class_dist = y_test.value_counts()
                    for idx, (class_label, count) in enumerate(class_dist.items()):
                        class_name = class_names[class_label]
                        percentage = (count / len(y_test)) * 100
                        st.metric(f"{class_name}", f"{count:,} ({percentage:.1f}%)")

            # TAB 2: Performance Metrics
            with tab2:
                st.markdown("### 📈 Model Performance Metrics")

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

                st.markdown("---")

                # Detailed metrics table
                st.markdown("### 📋 Detailed Metrics Table")
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

                st.markdown("---")

                # Metric explanations
                with st.expander("ℹ️ Understanding the Metrics"):
                    st.markdown("""
                    **Accuracy**: Overall percentage of correct predictions

                    **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes (higher is better)

                    **Precision**: Of all positive predictions, how many were actually correct

                    **Recall**: Of all actual positives, how many were correctly identified

                    **F1 Score**: Harmonic mean of precision and recall

                    **MCC (Matthews Correlation Coefficient)**: Balanced metric for imbalanced datasets (-1 to +1)
                    """)

            # TAB 3: Confusion Matrix
            with tab3:
                st.markdown("### 🔲 Confusion Matrix Visualization")

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
                    st.metric("Accuracy", f"{(total_correct/total_samples)*100:.2f}%")

                st.markdown("---")

                # Confusion matrix details
                st.markdown("### 🔍 Confusion Matrix Details")
                cm_df = pd.DataFrame(cm,
                                    index=[f"True {c}" for c in class_names],
                                    columns=[f"Pred {c}" for c in class_names])
                st.dataframe(cm_df, use_container_width=True)

            # TAB 4: Classification Report
            with tab4:
                st.markdown("### 📄 Detailed Classification Report")

                # Convert to DataFrame for better display
                report_df = pd.DataFrame(report).transpose()

                st.dataframe(
                    report_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', subset=['f1-score']),
                    use_container_width=True
                )

                st.markdown("---")

                # Per-class analysis
                st.markdown("### 📊 Per-Class Performance Analysis")
                for class_name in class_names:
                    if class_name in report:
                        st.markdown(f"#### Class: {class_name}")
                        class_metrics = report[class_name]

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Precision", f"{class_metrics['precision']:.4f}")
                        col2.metric("Recall", f"{class_metrics['recall']:.4f}")
                        col3.metric("F1-Score", f"{class_metrics['f1-score']:.4f}")
                        col4.metric("Support", f"{int(class_metrics['support'])}")

                        st.markdown("---")

            # TAB 5: Export Results
            with tab5:
                st.markdown("### 💾 Download Predictions and Results")

                # Predictions dataframe
                predictions_df = pd.DataFrame({
                    'True_Label': y_test.values,
                    'Predicted_Label': y_pred,
                    'True_Class': [class_names[i] for i in y_test.values],
                    'Predicted_Class': [class_names[i] for i in y_pred],
                    'Correct': y_test.values == y_pred
                })

                # Add prediction probabilities if available
                if y_scores is not None and hasattr(model, 'predict_proba'):
                    for idx, class_name in enumerate(class_names):
                        predictions_df[f'Prob_{class_name}'] = y_scores[:, idx]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📥 Download Predictions")
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
                        mime="text/csv"
                    )

                    st.markdown("**File includes:**")
                    st.markdown("- True and predicted labels")
                    st.markdown("- Correctness indicator")
                    if y_scores is not None and hasattr(model, 'predict_proba'):
                        st.markdown("- Prediction probabilities for each class")

                with col2:
                    st.markdown("#### 📊 Download Metrics Report")
                    metrics_report = f"""
Model Performance Report
========================
Model: {selected_model_name}
Dataset Size: {X_test.shape[0]} samples
Features: {X_test.shape[1]}

Metrics
-------
Accuracy: {metrics['Accuracy']:.4f}
AUC: {metrics['AUC']:.4f}
Precision: {metrics['Precision']:.4f}
Recall: {metrics['Recall']:.4f}
F1 Score: {metrics['F1 Score']:.4f}
MCC: {metrics['MCC']:.4f}

Per-Class Performance
--------------------
{pd.DataFrame(report).transpose().to_string()}
"""
                    st.download_button(
                        label="📥 Download Metrics Report",
                        data=metrics_report,
                        file_name=f"metrics_report_{selected_model_name.replace(' ', '_').lower()}.txt",
                        mime="text/plain"
                    )

                st.markdown("---")

                # Preview predictions
                st.markdown("### 👀 Preview Predictions")

                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    show_only_errors = st.checkbox("Show only misclassifications")
                with col2:
                    n_rows = st.slider("Number of rows to display", 5, 100, 20)

                if show_only_errors:
                    display_df = predictions_df[predictions_df['Correct'] == False].head(n_rows)
                    if len(display_df) == 0:
                        st.success("🎉 No misclassifications found!")
                    else:
                        st.dataframe(display_df, use_container_width=True)
                else:
                    st.dataframe(predictions_df.head(n_rows), use_container_width=True)

                st.markdown(f"**Total Misclassifications:** {(predictions_df['Correct'] == False).sum()} / {len(predictions_df)}")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.exception(e)

    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a test data CSV file to begin evaluation.")

        # Create tabs for home page
        tab1, tab2 = st.tabs(["📝 Instructions", "🏆 Model Comparison"])

        with tab1:
            st.markdown("### 📝 Expected File Format")
            st.markdown("""
            The test data CSV should contain:
            - All 51 feature columns (after preprocessing)
            - One target column: `failure_risk_label`
            - No missing values

            You can use the `test_data.csv` file included in the repository.
            """)

            st.markdown("### 🚀 Quick Start Guide")
            st.markdown("""
            1. **Upload Data**: Click the file uploader above and select `test_data.csv`
            2. **Select Model**: Choose from 5 trained classification models
            3. **Explore Results**: Navigate through tabs to view different aspects:
               - 📊 Overview: Quick summary and data preview
               - 📈 Performance Metrics: Detailed evaluation metrics
               - 🔲 Confusion Matrix: Visual representation of predictions
               - 📄 Classification Report: Per-class performance
               - 💾 Export Results: Download predictions and reports
            """)

        with tab2:
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
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']),
                use_container_width=True
            )

            st.markdown("**🏅 Best Overall Model:** Logistic Regression")
            st.markdown("Achieves the highest scores across all six evaluation metrics.")

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
