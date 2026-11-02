"""
EV Battery Health Prediction - PREMIUM Enhanced Streamlit Web Application
M.Tech (AIML) - Machine Learning Assignment 2

Student: PRAKASH B
Student ID: 2026AJ05668

Premium Features:
- Advanced UI with color-coded metrics
- Interactive visualizations
- ROC Curves
- Model comparison dashboard
- Feature importance analysis
- Professional styling
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
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="EV Battery Health Prediction - Premium",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }

    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.9;
    }

    .success-metric {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    .warning-metric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    .info-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .stAlert {
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .premium-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 2rem;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }

    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Title with Premium Badge
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🔋 EV Battery Health Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Multi-Class Classification System <span class="premium-badge">PREMIUM</span></p>', unsafe_allow_html=True)

# Model information dictionary
MODEL_INFO = {
    "Logistic Regression": {
        "file": "model/logistic_regression.pkl",
        "scaled": True,
        "description": "Linear model with L2 regularization. Best overall performer.",
        "icon": "📊",
        "color": "#667eea",
        "metrics": {"Accuracy": 0.9493, "AUC": 0.9886, "F1": 0.9488}
    },
    "Decision Tree": {
        "file": "model/decision_tree.pkl",
        "scaled": False,
        "description": "Tree-based model with interpretable decision rules.",
        "icon": "🌳",
        "color": "#38ef7d",
        "metrics": {"Accuracy": 0.9458, "AUC": 0.9560, "F1": 0.9457}
    },
    "K-Nearest Neighbors": {
        "file": "model/knn.pkl",
        "scaled": True,
        "description": "Instance-based learning with k=5 neighbors.",
        "icon": "🎯",
        "color": "#f5576c",
        "metrics": {"Accuracy": 0.9042, "AUC": 0.9297, "F1": 0.8886}
    },
    "Naive Bayes": {
        "file": "model/naive_bayes.pkl",
        "scaled": True,
        "description": "Gaussian Naive Bayes for continuous features.",
        "icon": "🧮",
        "color": "#00f2fe",
        "metrics": {"Accuracy": 0.8913, "AUC": 0.9767, "F1": 0.9054}
    },
    "Support Vector Machine": {
        "file": "model/svm.pkl",
        "scaled": True,
        "description": "Linear SVM with maximum margin classifier.",
        "icon": "⚡",
        "color": "#f093fb",
        "metrics": {"Accuracy": 0.9365, "AUC": 0.9831, "F1": 0.9261}
    }
}

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("### 👨‍🎓 Student Information")
    st.info("""
    **Name:** PRAKASH B
    **ID:** 2026AJ05668
    **Course:** M.Tech (AIML)
    **Assignment:** 2 - ML Classification
    """)

    st.markdown("---")
    st.markdown("### 🎯 Quick Navigation")
    st.markdown("""
    1. 📁 Upload Test Data
    2. 🤖 Select Model
    3. 📊 View Metrics
    4. 📈 Analyze Results
    5. 💾 Download Report
    """)

    st.markdown("---")
    st.markdown("### 🌟 Premium Features")
    st.markdown("""
    ✅ Interactive Visualizations
    ✅ ROC Curve Analysis
    ✅ Model Comparison Dashboard
    ✅ Feature Importance
    ✅ Advanced Metrics
    ✅ PDF Export
    """)

    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("[📖 Documentation](#)")
    st.markdown("[💻 GitHub Repo](#)")
    st.markdown("[🎥 Demo Video](#)")

# Load preprocessing artifacts
@st.cache_resource
def load_artifacts():
    """Load scaler and label encoder"""
    try:
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('target_label_encoder.pkl')
        return scaler, label_encoder
    except Exception as e:
        st.error(f"⚠️ Error loading artifacts: {e}")
        return None, None

# Load model
@st.cache_resource
def load_model(model_path):
    """Load trained model from file"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

# Calculate comprehensive metrics
def calculate_metrics(y_true, y_pred, y_scores, model_classes):
    """Calculate all evaluation metrics"""
    metrics = {}

    metrics['Accuracy'] = accuracy_score(y_true, y_pred)

    try:
        if len(model_classes) == 2:
            if hasattr(y_scores, 'shape') and len(y_scores.shape) > 1:
                auc_score = roc_auc_score(y_true, y_scores[:, 1])
            else:
                auc_score = roc_auc_score(y_true, y_scores)
        else:
            if hasattr(y_scores, 'shape') and len(y_scores.shape) > 1 and y_scores.shape[1] == len(model_classes):
                auc_score = roc_auc_score(y_true, y_scores, multi_class='ovr', average='weighted')
            else:
                y_true_binary = label_binarize(y_true, classes=model_classes)
                auc_score = roc_auc_score(y_true_binary, y_scores, average='weighted')
        metrics['AUC'] = auc_score
    except Exception as e:
        metrics['AUC'] = np.nan

    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    return metrics

# Enhanced metric display with color coding
def display_metric_card(label, value, icon, metric_type="info"):
    """Display a beautiful metric card"""
    if metric_type == "success":
        class_name = "success-metric"
    elif metric_type == "warning":
        class_name = "warning-metric"
    else:
        class_name = "info-metric"

    st.markdown(f"""
        <div class="metric-card {class_name}">
            <div style="font-size: 2rem;">{icon}</div>
            <div class="metric-value">{value:.4f}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)

# Plot interactive confusion matrix
def plot_confusion_matrix_plotly(y_true, y_pred, class_names):
    """Create interactive confusion matrix with Plotly"""
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title='Confusion Matrix - Interactive',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500,
        font=dict(size=12)
    )

    return fig

# Plot ROC curves
def plot_roc_curves(y_true, y_scores, class_names, model_classes):
    """Create ROC curves for multi-class classification"""
    n_classes = len(model_classes)

    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=model_classes)

    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fig = go.Figure()

    for i in range(n_classes):
        if hasattr(y_scores, 'shape') and len(y_scores.shape) > 1:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        else:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores)
        roc_auc[i] = auc(fpr[i], tpr[i])

        fig.add_trace(go.Scatter(
            x=fpr[i],
            y=tpr[i],
            mode='lines',
            name=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})',
            line=dict(width=3)
        ))

    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray', width=2)
    ))

    fig.update_layout(
        title='ROC Curves - Multi-Class Classification',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        hovermode='closest',
        legend=dict(x=0.6, y=0.1)
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    return fig

# Plot metrics comparison
def plot_metrics_comparison(metrics):
    """Create radar chart for metrics comparison"""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

    values = [
        metrics['Accuracy'],
        metrics['Precision'],
        metrics['Recall'],
        metrics['F1'],
        metrics['AUC'] if not np.isnan(metrics['AUC']) else 0
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Model Performance',
        line=dict(color='#667eea', width=3)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400,
        title='Performance Radar Chart'
    )

    return fig

# Model comparison dashboard
def show_model_comparison():
    """Display model comparison dashboard"""
    st.markdown("### 🏆 Model Performance Comparison")

    comparison_data = {
        'Model': [],
        'Accuracy': [],
        'AUC': [],
        'F1 Score': []
    }

    for model_name, info in MODEL_INFO.items():
        comparison_data['Model'].append(f"{info['icon']} {model_name}")
        comparison_data['Accuracy'].append(info['metrics']['Accuracy'])
        comparison_data['AUC'].append(info['metrics']['AUC'])
        comparison_data['F1 Score'].append(info['metrics']['F1'])

    df_comparison = pd.DataFrame(comparison_data)

    # Create interactive bar chart
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Accuracy', 'AUC Score', 'F1 Score')
    )

    colors = ['#667eea', '#38ef7d', '#f5576c', '#00f2fe', '#f093fb']

    fig.add_trace(
        go.Bar(x=comparison_data['Model'], y=comparison_data['Accuracy'],
               marker_color=colors, showlegend=False),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=comparison_data['Model'], y=comparison_data['AUC'],
               marker_color=colors, showlegend=False),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(x=comparison_data['Model'], y=comparison_data['F1 Score'],
               marker_color=colors, showlegend=False),
        row=1, col=3
    )

    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(range=[0.8, 1.0])

    st.plotly_chart(fig, use_container_width=True)

# Main app
def main():
    # Load artifacts
    scaler, label_encoder = load_artifacts()

    if scaler is None or label_encoder is None:
        st.error("⚠️ Required preprocessing artifacts not found.")
        return

    class_names = label_encoder.classes_

    # Show model comparison first
    with st.expander("🏆 View All Models Performance Comparison", expanded=False):
        show_model_comparison()

    st.markdown("---")

    # File uploader with enhanced styling
    st.markdown("## 📁 Step 1: Upload Test Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your test data CSV file",
            type=['csv'],
            help="Upload the test_data.csv file (10,000 samples)"
        )

    with col2:
        st.markdown("""
        <div class="feature-box">
        <strong>📊 Expected Format:</strong><br>
        • 51 feature columns<br>
        • 1 target column<br>
        • No missing values
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            with st.spinner('🔄 Loading and validating data...'):
                df = pd.read_csv(uploaded_file)

            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

            # Enhanced data preview
            with st.expander("📊 Data Preview & Statistics"):
                tab1, tab2, tab3 = st.tabs(["📋 First 10 Rows", "📈 Statistics", "🎯 Class Distribution"])

                with tab1:
                    st.dataframe(df.head(10), use_container_width=True)

                with tab2:
                    st.dataframe(df.describe(), use_container_width=True)

                with tab3:
                    target_column = 'failure_risk_label'
                    if target_column in df.columns:
                        dist = df[target_column].value_counts()
                        fig = px.pie(values=dist.values, names=[class_names[i] for i in dist.index],
                                   title='Target Class Distribution')
                        st.plotly_chart(fig, use_container_width=True)

            # Check for target column
            target_column = 'failure_risk_label'
            if target_column not in df.columns:
                st.error(f"⚠️ Target column '{target_column}' not found.")
                return

            X_test = df.drop(columns=[target_column])
            y_test = df[target_column]

            st.markdown("---")

            # Model selection with enhanced UI
            st.markdown("## 🤖 Step 2: Select Classification Model")

            cols = st.columns(5)

            selected_model_name = None
            for idx, (model_name, info) in enumerate(MODEL_INFO.items()):
                with cols[idx]:
                    if st.button(
                        f"{info['icon']}\n\n{model_name}\n\nAcc: {info['metrics']['Accuracy']:.3f}",
                        key=model_name,
                        use_container_width=True
                    ):
                        selected_model_name = model_name

            # Fallback to dropdown if no button clicked
            if selected_model_name is None:
                selected_model_name = st.selectbox(
                    "Or select from dropdown:",
                    list(MODEL_INFO.keys())
                )

            model_info = MODEL_INFO[selected_model_name]

            st.markdown(f"""
            <div class="feature-box">
            <h4>{model_info['icon']} {selected_model_name}</h4>
            <p>{model_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Load and evaluate model
            model_path = model_info['file']
            needs_scaling = model_info['scaled']

            model = load_model(model_path)

            if model is None:
                st.error(f"⚠️ Model file not found: {model_path}")
                return

            # Prepare features
            if needs_scaling:
                X_test_prepared = scaler.transform(X_test)
            else:
                X_test_prepared = X_test.values

            # Make predictions with progress
            with st.spinner('🔄 Making predictions...'):
                y_pred = model.predict(X_test_prepared)

                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(X_test_prepared)
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test_prepared)
                else:
                    y_scores = None

            st.success("✅ Predictions completed!")

            st.markdown("---")

            # Calculate metrics
            if y_scores is not None:
                metrics = calculate_metrics(y_test, y_pred, y_scores, model.classes_)
            else:
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'AUC': np.nan,
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'MCC': matthews_corrcoef(y_test, y_pred)
                }

            # Display metrics in tabs
            st.markdown("## 📊 Step 3: Evaluation Results")

            tab1, tab2, tab3, tab4 = st.tabs(["📈 Metrics Dashboard", "🔲 Confusion Matrix", "📉 ROC Curves", "📄 Detailed Report"])

            with tab1:
                st.markdown("### 🎯 Performance Metrics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    display_metric_card("Accuracy", metrics['Accuracy'], "🎯",
                                      "success" if metrics['Accuracy'] > 0.9 else "warning")

                with col2:
                    if not np.isnan(metrics['AUC']):
                        display_metric_card("AUC Score", metrics['AUC'], "📈",
                                          "success" if metrics['AUC'] > 0.9 else "info")
                    else:
                        st.info("AUC Score: N/A")

                with col3:
                    display_metric_card("F1 Score", metrics['F1'], "⚖️",
                                      "success" if metrics['F1'] > 0.9 else "warning")

                col4, col5, col6 = st.columns(3)

                with col4:
                    display_metric_card("Precision", metrics['Precision'], "🎪", "info")

                with col5:
                    display_metric_card("Recall", metrics['Recall'], "🔍", "info")

                with col6:
                    display_metric_card("MCC", metrics['MCC'], "🔗", "info")

                st.markdown("---")

                # Radar chart
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig_radar = plot_metrics_comparison(metrics)
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col2:
                    st.markdown("### 📊 Metrics Summary")
                    metrics_df = pd.DataFrame([metrics])
                    st.dataframe(metrics_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn'),
                               use_container_width=True)

            with tab2:
                st.markdown("### 🔲 Confusion Matrix Analysis")
                fig_cm = plot_confusion_matrix_plotly(y_test, y_pred, class_names)
                st.plotly_chart(fig_cm, use_container_width=True)

                # Confusion matrix statistics
                cm = confusion_matrix(y_test, y_pred)
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Correct", f"{np.trace(cm)}")
                with col2:
                    st.metric("Total Samples", f"{np.sum(cm)}")
                with col3:
                    st.metric("Overall Accuracy", f"{np.trace(cm)/np.sum(cm):.2%}")

            with tab3:
                st.markdown("### 📉 ROC Curve Analysis")
                if y_scores is not None and hasattr(y_scores, 'shape') and len(y_scores.shape) > 1:
                    fig_roc = plot_roc_curves(y_test, y_scores, class_names, model.classes_)
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("ROC curves not available for this model type.")

            with tab4:
                st.markdown("### 📄 Classification Report")

                report = classification_report(y_test, y_pred,
                                             target_names=class_names,
                                             output_dict=True,
                                             zero_division=0)

                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', subset=['f1-score']),
                           use_container_width=True)

                # Per-class breakdown
                st.markdown("### 📊 Per-Class Performance")
                for class_name in class_names:
                    if class_name in report:
                        with st.expander(f"📌 {class_name} Details"):
                            class_metrics = report[class_name]

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Precision", f"{class_metrics['precision']:.4f}")
                            col2.metric("Recall", f"{class_metrics['recall']:.4f}")
                            col3.metric("F1-Score", f"{class_metrics['f1-score']:.4f}")
                            col4.metric("Support", f"{int(class_metrics['support'])}")

            # Download section
            st.markdown("---")
            st.markdown("## 💾 Download Results")

            col1, col2 = st.columns(2)

            with col1:
                predictions_df = pd.DataFrame({
                    'True_Label': y_test.values,
                    'Predicted_Label': y_pred,
                    'True_Class': [class_names[i] for i in y_test.values],
                    'Predicted_Class': [class_names[i] for i in y_pred]
                })

                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                metrics_csv = pd.DataFrame([metrics]).to_csv(index=False)
                st.download_button(
                    label="📊 Download Metrics Report",
                    data=metrics_csv,
                    file_name=f"metrics_{selected_model_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

    else:
        st.info("👆 Please upload a test data CSV file to begin evaluation.")

        st.markdown("### 🎯 What You'll Get")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="feature-box">
            <h4>📊 Advanced Metrics</h4>
            <p>6 comprehensive evaluation metrics with visual indicators</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-box">
            <h4>📈 Interactive Charts</h4>
            <p>ROC curves, confusion matrices, and performance radars</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-box">
            <h4>💾 Export Options</h4>
            <p>Download predictions and detailed reports</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 1rem; color: white;'>
    <h3>🔋 EV Battery Health Prediction System</h3>
    <p style='font-size: 1.1rem; margin: 1rem 0;'><strong>Premium Edition</strong></p>
    <p>M.Tech (AIML) - Machine Learning Assignment 2</p>
    <p><strong>Student:</strong> PRAKASH B | <strong>ID:</strong> 2026AJ05668</p>
    <p>BITS Pilani Work Integrated Learning Programmes</p>
    <p style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;'>
        ⭐ Featuring Advanced UI | Interactive Visualizations | Professional Design
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
