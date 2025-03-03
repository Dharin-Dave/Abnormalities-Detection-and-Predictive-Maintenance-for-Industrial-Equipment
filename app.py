import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

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
    
    return rf_model, lr_model

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
        # For Random Forest
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

# Load data using the uploaded file
df = load_data(uploaded_file)

if df is not None:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    # Replace radio buttons with tabs
    tabs = st.tabs(["Data Overview", "Exploratory Analysis", "Model Performance", "Prediction"])
    
    # Process data
    X_train, X_test, y_train, y_test, scaler, num_cols, feature_names = process_data(df)
    
    # Train models
    rf_model, lr_model = train_models(X_train, y_train)
    
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
        fig = px.bar(df['location'].value_counts().reset_index(), 
                    x='index', y='location', 
                    labels={'index': 'Location', 'location': 'Count'},
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
        
        # Distribution of numerical features
        st.subheader("Distribution of Numerical Features")
        
        feature = st.selectbox("Select Feature", 
                               ["temperature", "pressure", "vibration", "humidity"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=feature, color="faulty", 
                               marginal="box", 
                               title=f"Distribution of {feature}",
                               color_discrete_sequence=["blue", "red"])
            st.plotly_chart(fig)
        
        with col2:
            # Boxplot by equipment
            fig = px.box(df, x="equipment", y=feature, color="faulty",
                         title=f"{feature} by Equipment Type",
                         color_discrete_sequence=["blue", "red"])
            st.plotly_chart(fig)
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr = df[["temperature", "pressure", "vibration", "humidity", "faulty"]].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig)
        
        # Scatter plots
        st.subheader("Feature Relationships")
        x_axis = st.selectbox("X-axis", ["temperature", "pressure", "vibration", "humidity"], key="x_axis")
        y_axis = st.selectbox("Y-axis", ["pressure", "temperature", "vibration", "humidity"], key="y_axis")
        
        fig = px.scatter(df, x=x_axis, y=y_axis, color="faulty", 
                         facet_col="equipment", 
                         title=f"{y_axis} vs {x_axis} by Equipment Type",
                         color_discrete_sequence=["blue", "red"])
        st.plotly_chart(fig)
        
        # Feature distribution by faulty status
        st.subheader("Feature Distribution by Faulty Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x="temperature", color="faulty", barmode="group",
                              title="Temperature Distribution by Faulty Status")
            st.plotly_chart(fig)
            
            fig = px.histogram(df, x="vibration", color="faulty", barmode="group",
                              title="Vibration Distribution by Faulty Status")
            st.plotly_chart(fig)
        
        with col2:
            fig = px.histogram(df, x="pressure", color="faulty", barmode="group",
                              title="Pressure Distribution by Faulty Status")
            st.plotly_chart(fig)
            
            fig = px.histogram(df, x="humidity", color="faulty", barmode="group",
                              title="Humidity Distribution by Faulty Status")
            st.plotly_chart(fig)
    
    # Model performance
    with tabs[2]:
        st.header("Model Performance")
        
        # Select model
        model_option = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
        
        if model_option == "Random Forest":
            model = rf_model
            model_name = "Random Forest"
        else:
            model = lr_model
            model_name = "Logistic Regression"
        
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
        st.header("Predict Equipment Failure")
        st.write("Enter equipment parameters to predict if it's likely to fail.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider("Temperature", float(df['temperature'].min()), float(df['temperature'].max()), float(df['temperature'].mean()))
            pressure = st.slider("Pressure", float(df['pressure'].min()), float(df['pressure'].max()), float(df['pressure'].mean()))
        
        with col2:
            vibration = st.slider("Vibration", float(df['vibration'].min()), float(df['vibration'].max()), float(df['vibration'].mean()))
            humidity = st.slider("Humidity", float(df['humidity'].min()), float(df['humidity'].max()), float(df['humidity'].mean()))
        
        equipment = st.selectbox("Equipment Type", df['equipment'].unique())
        location = st.selectbox("Location", df['location'].unique())
        
        # Create a dataframe for the input
        input_data = pd.DataFrame({
            'temperature': [temperature],
            'pressure': [pressure],
            'vibration': [vibration],
            'humidity': [humidity],
            'equipment': [equipment],
            'location': [location]
        })
        
        # One-hot encode the input
        input_encoded = pd.get_dummies(input_data, drop_first=True)
        
        # Ensure all columns from training are present
        for col in X_train.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[X_train.columns]
        
        # Select model BEFORE the predict button
        model_option = st.radio("Select Model for Prediction", 
                               ["Random Forest", "Logistic Regression"], 
                               key="prediction_model_selector")
        
        # Make prediction
        if st.button("Predict"):
            if model_option == "Random Forest":
                model = rf_model
                model_name = "Random Forest"
            else:  # Fix for Logistic Regression
                model = lr_model
                model_name = "Logistic Regression"
            
            # Predict
            prediction = model.predict(input_encoded)[0]
            probability = model.predict_proba(input_encoded)[0][1]
            
            # Display result
            st.subheader("Prediction Result")
            if prediction:
                st.error(f"⚠️ Equipment is predicted to be FAULTY with {probability:.2%} probability")
            else:
                st.success(f"✅ Equipment is predicted to be NORMAL with {(1-probability):.2%} probability")
            
            # Gauge chart for failure probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Failure Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100
                    }
                }
            ))
            st.plotly_chart(fig)
            
            # Explanation based on feature importance
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                st.subheader("Prediction Explanation")
                
                if hasattr(model, 'feature_importances_'):
                    # For Random Forest
                    importances = model.feature_importances_
                    feature_importance = dict(zip(X_train.columns, importances))
                elif hasattr(model, 'coef_'):
                    # For Logistic Regression
                    coefficients = model.coef_[0]
                    feature_importance = dict(zip(X_train.columns, coefficients))
                
                # Sort by absolute importance/coefficient
                sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                
                # Display top 5 features
                st.write("Top 5 influential features for this prediction:")
                for feature, importance in sorted_importance[:5]:
                    st.write(f"- {feature}: {importance:.4f}")
else:
    st.info("Please upload the equipment_anomaly_data.csv file to start the analysis.")