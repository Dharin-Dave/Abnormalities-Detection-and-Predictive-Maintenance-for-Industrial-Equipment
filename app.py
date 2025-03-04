import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Industrial Equipment Abnormality Detection",
    page_icon="🏭",
    layout="wide"
)

# Application title and introduction
st.title("Industrial Equipment Abnormality Detection and Predictive Maintenance")
st.markdown("""
This dashboard analyzes industrial equipment data to detect abnormalities and predict maintenance needs.
The system uses machine learning to identify patterns that indicate potential equipment failures.
""")

# Move file uploader outside the cached function
uploaded_file = st.file_uploader("Upload equipment_anomaly_data.csv", type="csv")

# Load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['faulty'] = df['faulty'].astype(bool)
        return df
    else:
        return None

# Data processing function
@st.cache_data
def process_data(df):
    # Create feature columns and target variable
    X = pd.get_dummies(df.drop(['faulty'], axis=1), drop_first=True)
    y = df['faulty']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the numerical features
    scaler = StandardScaler()
    num_cols = ['temperature', 'pressure', 'vibration', 'humidity']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, num_cols, X.columns

# Model training and evaluation functions
@st.cache_resource
def train_models(X_train, y_train):
    # Random Forest model
    rf_model = RandomForestClassifier(max_depth=10, max_features=20, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Logistic Regression model
    lr_model = LogisticRegression(C=29.763514416313132, solver='liblinear', random_state=42)
    lr_model.fit(X_train, y_train)
    
    # XGBoost model
    xgb_model = XGBClassifier(objective="binary:logistic", random_state=42)
    xgb_model.fit(X_train, y_train)
    
    return rf_model, lr_model, xgb_model

def evaluate_model(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return report, fpr, tpr, roc_auc, cm, y_pred

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        # For Random Forest and XGBoost
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance ({model_name})')
        return fig
    elif hasattr(model, 'coef_'):
        # For Logistic Regression
        coefficients = model.coef_[0]
        indices = np.argsort(np.abs(coefficients))[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.bar(range(len(coefficients)), coefficients[indices], align='center')
        plt.xticks(range(len(coefficients)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        plt.title(f'Feature Coefficients ({model_name})')
        return fig
    return None

def detect_outliers(df, threshold=3):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    z_scores = stats.zscore(df[numeric_columns])
    outliers = (abs(z_scores) > threshold).any(axis=1)
    return df[outliers], z_scores

def detect_anomalies_isolation_forest(df, contamination=0.1):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    clf = IsolationForest(contamination=contamination, random_state=42)
    anomalies = clf.fit_predict(df[numeric_columns])
    return df[anomalies == -1], clf.decision_function(df[numeric_columns])

# Load data using the uploaded file
df = load_data(uploaded_file)

if df is not None:
    # Replace radio buttons with tabs
    tabs = st.tabs(["Data Overview", "Exploratory Analysis", "Model Performance", "Anomaly Detection"])
    
    # Process data
    X_train, X_test, y_train, y_test, scaler, num_cols, feature_names = process_data(df)
    
    # Train models
    rf_model, lr_model, xgb_model = train_models(X_train, y_train)
    
    # Data overview
    with tabs[0]:
        st.header("Data Overview")
        
        # Display dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Information")
            st.write(f"Number of records: {df.shape[0]}")
            st.write(f"Number of features: {df.shape[1] - 1}")  # Excluding target variable
            st.write(f"Number of faulty equipment: {df['faulty'].sum()}")
            st.write(f"Percentage of faulty equipment: {(df['faulty'].mean() * 100):.2f}%")
        
        with col2:
            st.subheader("Equipment Distribution")
            fig = px.pie(df, names='equipment', title='Equipment Types')
            st.plotly_chart(fig)
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
        # Display basic statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Location distribution
        st.subheader("Location Distribution")
        location_counts = df['location'].value_counts().reset_index()
        location_counts.columns = ['location', 'count']  # Explicitly name the columns
        fig = px.bar(location_counts, 
                    x='location', y='count', 
                    labels={'location': 'Location', 'count': 'Count'},
                    title='Count by Location')
        st.plotly_chart(fig)
        
        # Faulty distribution
        st.subheader("Faulty Equipment Distribution")
        fig = px.pie(df, names='faulty', title='Faulty vs Non-Faulty Equipment',
                    color='faulty', color_discrete_map={True: 'red', False: 'green'})
        st.plotly_chart(fig)
    
    # Exploratory analysis
    with tabs[1]:
        st.header("Exploratory Data Analysis")

        # Equipment Distribution
        st.subheader("Equipment Distribution")
        df_equip_counts = df['equipment'].value_counts()
        
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'bar'}, {'type':'pie'}]])
        fig.add_trace(go.Bar(x=df_equip_counts.index, y=df_equip_counts.values, name='Equipment Count'), row=1, col=1)
        fig.add_trace(go.Pie(labels=df_equip_counts.index, values=df_equip_counts.values, name='Equipment Percentage'), row=1, col=2)
        fig.update_layout(height=500, width=800, title_text="Equipment Distribution")
        st.plotly_chart(fig)

        # Location Distribution
        st.subheader("Location Distribution")
        df_location_counts = df['location'].value_counts()
        
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'bar'}, {'type':'pie'}]])
        fig.add_trace(go.Bar(x=df_location_counts.index, y=df_location_counts.values, name='Location Count'), row=1, col=1)
        fig.add_trace(go.Pie(labels=df_location_counts.index, values=df_location_counts.values, name='Location Percentage'), row=1, col=2)
        fig.update_layout(height=500, width=800, title_text="Location Distribution")
        st.plotly_chart(fig)

        # Equipment per Location
        st.subheader("Equipment per Location")
        df_location_equip_counts = df.pivot_table(index=['location'], columns='equipment', aggfunc='size', fill_value=0)
        fig = px.bar(df_location_equip_counts, barmode='group')
        fig.update_layout(height=500, width=800, title_text="Equipment per Location")
        st.plotly_chart(fig)

        # Temperature Distribution
        st.subheader("Temperature Distribution")
        fig = make_subplots(rows=1, cols=3)
        fig.add_trace(go.Histogram(x=df['temperature'], name='Overall', histnorm='percent'), row=1, col=1)
        for equipment in df['equipment'].unique():
            fig.add_trace(go.Histogram(x=df[df['equipment'] == equipment]['temperature'], name=equipment, histnorm='percent'), row=1, col=2)
        for location in df['location'].unique():
            fig.add_trace(go.Histogram(x=df[df['location'] == location]['temperature'], name=location, histnorm='percent'), row=1, col=3)
        fig.update_layout(height=500, width=1000, title_text="Temperature Distribution")
        st.plotly_chart(fig)

        # Pressure Distribution
        st.subheader("Pressure Distribution")
        fig = make_subplots(rows=1, cols=3)
        fig.add_trace(go.Histogram(x=df['pressure'], name='Overall', histnorm='percent'), row=1, col=1)
        for equipment in df['equipment'].unique():
            fig.add_trace(go.Histogram(x=df[df['equipment'] == equipment]['pressure'], name=equipment, histnorm='percent'), row=1, col=2)
        for location in df['location'].unique():
            fig.add_trace(go.Histogram(x=df[df['location'] == location]['pressure'], name=location, histnorm='percent'), row=1, col=3)
        fig.update_layout(height=500, width=1000, title_text="Pressure Distribution")
        st.plotly_chart(fig)

        # Vibration Distribution
        st.subheader("Vibration Distribution")
        fig = make_subplots(rows=1, cols=3)
        fig.add_trace(go.Histogram(x=df['vibration'], name='Overall', histnorm='percent'), row=1, col=1)
        for equipment in df['equipment'].unique():
            fig.add_trace(go.Histogram(x=df[df['equipment'] == equipment]['vibration'], name=equipment, histnorm='percent'), row=1, col=2)
        for location in df['location'].unique():
            fig.add_trace(go.Histogram(x=df[df['location'] == location]['vibration'], name=location, histnorm='percent'), row=1, col=3)
        fig.update_layout(height=500, width=1000, title_text="Vibration Distribution")
        st.plotly_chart(fig)

        # Humidity Distribution
        st.subheader("Humidity Distribution")
        fig = make_subplots(rows=1, cols=3)
        fig.add_trace(go.Histogram(x=df['humidity'], name='Overall', histnorm='percent'), row=1, col=1)
        for equipment in df['equipment'].unique():
            fig.add_trace(go.Histogram(x=df[df['equipment'] == equipment]['humidity'], name=equipment, histnorm='percent'), row=1, col=2)
        for location in df['location'].unique():
            fig.add_trace(go.Histogram(x=df[df['location'] == location]['humidity'], name=location, histnorm='percent'), row=1, col=3)
        fig.update_layout(height=500, width=1000, title_text="Humidity Distribution")
        st.plotly_chart(fig)

        # Fault Distribution by Location and Equipment
        st.subheader("Fault Distribution")
        df_location_faulty_counts = df.pivot_table(index=['location'], columns='faulty', aggfunc='size', fill_value=0)
        df_equip_faulty_counts = df.pivot_table(index=['equipment'], columns='faulty', aggfunc='size', fill_value=0)
        
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Bar(x=df_location_faulty_counts.index, y=df_location_faulty_counts[False], name='Not Faulty', offsetgroup=0), row=1, col=1)
        fig.add_trace(go.Bar(x=df_location_faulty_counts.index, y=df_location_faulty_counts[True], name='Faulty', offsetgroup=1), row=1, col=1)
        fig.add_trace(go.Bar(x=df_equip_faulty_counts.index, y=df_equip_faulty_counts[False], name='Not Faulty', offsetgroup=0), row=1, col=2)
        fig.add_trace(go.Bar(x=df_equip_faulty_counts.index, y=df_equip_faulty_counts[True], name='Faulty', offsetgroup=1), row=1, col=2)
        fig.update_layout(height=500, width=1000, title_text="Fault Distribution by Location and Equipment", barmode='group')
        st.plotly_chart(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        df_corr = df.drop(['equipment', 'location'], axis=1)
        correlation = df_corr.corr()
        fig = px.imshow(correlation, text_auto=True, aspect="auto")
        fig.update_layout(height=600, width=800, title_text="Correlation Heatmap")
        st.plotly_chart(fig)

        # Distribution of Measurements by Fault Status
        st.subheader("Distribution of Measurements by Fault Status")
        measurements = ['vibration', 'temperature', 'humidity', 'pressure']
        fig = make_subplots(rows=2, cols=2, subplot_titles=measurements)
        
        for i, measurement in enumerate(measurements):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(go.Histogram(x=df[df['faulty'] == False][measurement], name='Not Faulty', histnorm='percent'), row=row, col=col)
            fig.add_trace(go.Histogram(x=df[df['faulty'] == True][measurement], name='Faulty', histnorm='percent'), row=row, col=col)
        
        fig.update_layout(height=800, width=1000, title_text="Distribution of Measurements by Fault Status")
        st.plotly_chart(fig)
    
    # Model performance
    with tabs[2]:
        st.header("Model Performance")
        
        # Select model
        model_option = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "XGBoost"])

        if model_option == "Random Forest":
            model = rf_model
            model_name = "Random Forest"
        elif model_option == "Logistic Regression":
            model = lr_model
            model_name = "Logistic Regression"
        else:
            model = xgb_model
            model_name = "XGBoost"
        
        # Evaluate model
        report, fpr, tpr, roc_auc, cm, y_pred = evaluate_model(model, X_test, y_test, model_name)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {model_name}')
            st.pyplot(fig)
        
        with col2:
            st.subheader("ROC Curve")
            fig = px.area(
                x=fpr, y=tpr,
                title=f'ROC Curve (AUC={roc_auc:.4f}) - {model_name}',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            st.plotly_chart(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            fig = plot_feature_importance(model, feature_names, model_name)
            if fig:
                st.pyplot(fig)
            else:
                st.write("Feature importance not available for this model.")
    
    # Prediction
    with tabs[3]:
        st.header("Anomaly Detection")
        method = st.radio("Select Anomaly Detection Method", ["Outliers", "Isolation Forest"])
        
        if method == "Outliers":
            threshold = st.slider("Z-score Threshold", 2.0, 5.0, 3.0, 0.1)
            anomalies, scores = detect_outliers(df, threshold)
        else:
            contamination = st.slider("Contamination Factor", 0.01, 0.5, 0.1, 0.01)
            anomalies, scores = detect_anomalies_isolation_forest(df, contamination)
        
        st.subheader(f"Detected Anomalies ({len(anomalies)} rows)")
        
        if not anomalies.empty:
            # Create a styled dataframe
            def highlight_anomalies(row):
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                if method == "Outliers":
                    return ['background-color: red' if col in numeric_columns and abs(scores.loc[row.name, col]) > threshold else '' for col in row.index]
                else:
                    return ['background-color: red' if col in numeric_columns and scores[row.name] < np.percentile(scores, 10) else '' for col in row.index]
            styled_anomalies = anomalies.style.apply(highlight_anomalies, axis=1)
            st.dataframe(styled_anomalies)
        else:
            st.write("No anomalies detected.")
        
        # Visualize anomalies
        if not anomalies.empty:
            st.subheader("Anomaly Visualization")
            feature_x = st.selectbox("Select X-axis feature", df.select_dtypes(include=[np.number]).columns)
            feature_y = st.selectbox("Select Y-axis feature", df.select_dtypes(include=[np.number]).columns)
            
            fig = px.scatter(df, x=feature_x, y=feature_y, color=df.index.isin(anomalies.index),
                            color_discrete_map={True: 'red', False: 'blue'},
                            labels={'color': 'Is Anomaly'})
            st.plotly_chart(fig)
else:
    st.info("Please upload the equipment_anomaly_data.csv file to start the analysis.")