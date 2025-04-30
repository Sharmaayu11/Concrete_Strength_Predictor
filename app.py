import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import tempfile
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Custom CSS for enhanced UI
def load_css():
    css = """
    <style>
        body {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
            background-color: rgba(255, 255, 255, 0.97);
            border-radius: 20px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        }
        .title {
            color: #ffffff;
            text-align: center;
            font-size: 3.2em;
            font-weight: 800;
            margin-bottom: 0.3em;
            text-transform: uppercase;
            letter-spacing: 2px;
            background: linear-gradient(135deg, #00c4b4, #7b61ff);
            padding: 50px;
            border-radius: 15px;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
        }
        .subtitle {
            color: #111111;
            text-align: center;
            font-size: 1.5em;
            margin-bottom: 2.5em;
            font-style: italic;
            background-color: rgba(0, 188, 212, 0.2);
            padding: 12px;
            border-radius: 10px;
        }
        .how-to-use {
            background: linear-gradient(135deg, #e0f7fa, #80deea);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            border: 2px solid #00acc1;
            margin-bottom: 30px;
            font-size: 1.2em;
            line-height: 1.7;
            transition: all 0.3s ease;
            
        }
        .how-to-use:hover {
            box-shadow: 0 12px 30px rgba(0,0,0,0.25);
            transform: translateY(-5px);
        }
        .how-to-use h3 {
            color: #7b61ff;
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 25px;
            text-align: center;
            text-transform: uppercase;
        }
        .how-to-use li {
            margin-bottom: 20px;
            padding-left: 35px;
            position: relative;
            transition: transform 0.2s ease, color 0.2s ease;
        }
        .how-to-use li:hover {
            transform: translateX(10px);
            color: #00796b;
        }
        .how-to-use li:before {
            content: '✅';
            position: absolute;
            left: 0;
            font-size: 1.3em;
        }
        .how-to-use hr {
            border: 0;
            height: 1px;
            background: linear-gradient(to right, transparent, #00acc1, transparent);
            margin: 20px 0;
        }
        .stExpander summary {
            color: #111111 !important;
            font-weight: 800 !important;
            font-size: 3em !important;
        }
        .metric-card {
            background: linear-gradient(45deg, #ffffff, #e6f3fa);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.3);
        }
        .metric-card h3 {
            color: #00695c;
            font-size: 1.8em;
            margin-bottom: 15px;
            font-weight: 600;
        }
        .metric-card p {
            color: #f4511e;
            font-size: 2.4em;
            font-weight: 700;
        }
        .input-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(240,247,255,0.95));
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .stButton>button {
            background: linear-gradient(45deg, #f4511e, #ff8a65);
            color: white;
            border-radius: 30px;
            padding: 14px 40px;
            font-size: 1.3em;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #d84315, #f06292);
            transform: scale(1.08);
        }
        .prediction-result {
            font-size: 2em;
            font-weight: 700;
            color: #00695c;
            text-align: center;
            margin-top: 40px;
            background-color: #e0f7fa;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #00c4b4, #7b61ff);
            padding: 25px;
            border-radius: 15px;
            background-color: rgba(255,255,255,0.95);
        }
        .stNumberInput input {
            border-radius: 10px;
            padding: 12px;
            font-size: 1.2em;
            border: 2px solid #00acc1;
            background-color: #f0f7ff;
            color: #00695c;
        }
        .stNumberInput label {
            color: #00695c;
            font-weight: 500;
            font-size: 1.2em;
        }
        .stSelectbox select {
            border-radius: 10px;
            padding: 12px;
            font-size: 1.2em;
            background-color: #f0f7ff;
            border: 2px solid #00acc1;
        }
        .sidebar .stFileUploader {
            background-color: #f0f7ff;
            border-radius: 12px;
            padding: 20px;
            border: 2px solid #00acc1;
        }
        .stPlotlyChart {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .stApp h2 {
            color: #00695c;
            font-weight: 700;
            border-bottom: 3px solid #00acc1;
            padding-bottom: 12px;
        }
        .stApp h3 {
            color: #00695c;
            font-weight: 600;
            border-bottom: 2px solid #00acc1;
            padding-bottom: 10px;
        }
        .credit-text {
            color: #ffffff;
            font-size: 1.2em;
            font-weight: 500;
            text-align: center;
            margin-top: 30px;
        }
        .hyperparam-text {
            color: #7b61ff;
            font-size: 1.4em;
            font-weight: 600;
        }
        .stSpinner > div {
            background-color: #2D2D2D;
            color: #E0E0E0;
            padding: 10px;
            border-radius: 8px;
            font-size: 1.2em;
            font-weight: 500;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .stError {
            background-color: #ffebee;
            border: 2px solid #ef5350;
            border-radius: 10px;
            padding: 15px;
            font-size: 1.2em;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load Google Fonts
def load_fonts():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# Load and preprocess the dataset
@st.cache_data
def load_data(file_path, ash_type):
    try:
        data = pd.read_csv(file_path)
        ash_column = "Fly ash (kg/m3)" if ash_type == "Flyash" else "Rice husk ash (kg/m3)"
        required_columns = [
            "Water (kg/m3)", "Cement (kg/m3)", "Fine aggregate (kg/m3)",
            "Coarse aggregate (kg/m3)", ash_column, "Age (days)",
            "Super plasticizer (kg/m3)", "Compressive strength (MPa)"
        ]
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"CSV missing required columns: {', '.join(missing_cols)}")
        if not data[required_columns].select_dtypes(include=[np.number]).columns.equals(data[required_columns].columns):
            non_numeric_cols = [col for col in required_columns if not pd.api.types.is_numeric_dtype(data[col])]
            raise ValueError(f"CSV contains non-numeric data in columns: {', '.join(non_numeric_cols)}")
        if data[required_columns].isna().any().any():
            raise ValueError("CSV contains missing values")
        if len(data) < 10:
            raise ValueError("CSV has too few rows (minimum 10 required)")
        
        X = data[required_columns[:-1]]
        y = data["Compressive strength (MPa)"]
        logger.info(f"Dataset loaded: shape={X.shape}, columns={list(X.columns)}")
        return X, y, data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        st.error(f"Error loading dataset: {e}")
        return None, None, None

# Load saved model based on ash type
def load_model(ash_type):
    try:
        model_path = "flyash_model.pkl" if ash_type == "Flyash" else "rha_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found in {os.getcwd()}")
        model = joblib.load(model_path)
        logger.info(f"{ash_type} model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading {ash_type} model: {e}")
        st.error(f"Error loading {ash_type} model: {e}. Please ensure the model file exists and is compatible.")
        return None

# Train XGBoost model (used only for uploaded datasets)
@st.cache_resource
def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
        ])
        param_grid = {
            'xgb__n_estimators': [300, 400, 500, 600, 700],
            'xgb__max_depth': [3, 5, 7, 9, 11],
            'xgb__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'xgb__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'xgb__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'xgb__gamma': [0, 0.1, 0.2, 0.3],
            'xgb__reg_alpha': [0, 0.001, 0.005, 0.01],
            'xgb__reg_lambda': [1, 1.2, 1.5, 2.0]
        }
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=50, scoring='neg_root_mean_squared_error',
            cv=5, n_jobs=-1, verbose=1, random_state=42
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = -search.cv_results_['mean_test_score']
        cv_mean_rmse = cv_scores[search.best_index_]
        cv_std_rmse = search.cv_results_['std_test_score'][search.best_index_]
        logger.info(f"Model trained: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.2f}, CV_RMSE={cv_mean_rmse:.2f}±{cv_std_rmse:.2f}")
        return best_model, mse, mae, r2, X_test, y_test, y_pred, search.best_params_, cv_mean_rmse, cv_std_rmse
    except Exception as e:
        logger.error(f"Error training model: {e}")
        st.error(f"Error training model: {e}")
        return None, None, None, None, None, None, None, None, None, None

# Predict compressive strength
def predict_strength(model, new_data, ash_type):
    try:
        new_data = np.array(new_data).reshape(1, -1)
        if new_data.shape[1] != 7:
            raise ValueError(f"Input must contain exactly 7 features, got {new_data.shape[1]}")
        prediction = model.predict(new_data)[0]
        logger.info(f"Prediction for {ash_type}: {prediction:.2f} MPa")
        return prediction
    except Exception as e:
        logger.error(f"Error in prediction for {ash_type}: {e}")
        st.error(f"Prediction failed: {e}. Please check input values or model compatibility.")
        return None

# Save model based on ash type
def save_model(model, ash_type):
    try:
        model_path = "flyash_model.pkl" if ash_type == "Flyash" else "rha_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"{ash_type} model saved as {model_path}")
        st.success(f"{ash_type} model saved successfully as {model_path}")
    except Exception as e:
        logger.error(f"Error saving {ash_type} model: {e}")
        st.error(f"Error saving {ash_type} model: {e}")

# Generate plots for uploaded dataset
def generate_plots(data, X_test, y_test, y_pred, model, ash_type):
    plots = []
    
    # 1. Scatter Plot (Cement vs Strength)
    try:
        scatter_fig = px.scatter(
            data,
            x='Cement (kg/m3)',
            y='Compressive strength (MPa)',
            color='Water (kg/m3)',
            title="Cement vs Strength (Colored by Water Content)",
            labels={'Cement (kg/m3)': 'Cement (kg/m³)', 'Compressive strength (MPa)': 'Strength (MPa)'},
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        scatter_fig.update_layout(title_font_size=20, title_x=0.5, margin=dict(l=20, r=20, t=50, b=20))
        plots.append(("Cement vs Strength", scatter_fig))
    except Exception as e:
        logger.error(f"Error rendering scatter plot: {e}")
        st.error(f"Error rendering scatter plot: {e}")
    
    # 2. Pair Plot
    try:
        pair_fig = px.scatter_matrix(
            data,
            dimensions=data.columns[:-1],
            color='Compressive strength (MPa)',
            title="Pair Plot of Features vs Strength",
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        pair_fig.update_layout(title_font_size=20, title_x=0.5, height=800)
        plots.append(("Pair Plot", pair_fig))
    except Exception as e:
        logger.error(f"Error rendering pair plot: {e}")
        st.error(f"Error rendering pair plot: {e}")
    
    # 3. Correlation Heatmap
    try:
        corr_matrix = data.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        fig.update_layout(title="Correlation Heatmap", title_font_size=20, title_x=0.5, width=600, height=600)
        plots.append(("Correlation Heatmap", fig))
    except Exception as e:
        logger.error(f"Error rendering heatmap: {e}")
        st.error(f"Error rendering heatmap: {e}")
    
    # 4. Distribution Plots
    try:
        for col in data.columns:
            dist_fig = px.histogram(
                data,
                x=col,
                title=f"Distribution of {col}",
                template='plotly_white',
                color_discrete_sequence=['#00695c'],
                nbins=30
            )
            dist_fig.update_layout(title_font_size=20, title_x=0.5, bargap=0.2)
            plots.append((f"Distribution of {col}", dist_fig))
    except Exception as e:
        logger.error(f"Error rendering distribution plots: {e}")
        st.error(f"Error rendering distribution plots: {e}")
    
    # 5. Residual Plot
    try:
        residuals = y_test - y_pred
        resid_fig = px.scatter(
            x=y_pred,
            y=residuals,
            labels={'x': 'Predicted (MPa)', 'y': 'Residuals (MPa)'},
            title="Residuals vs Predicted",
            template='plotly_white'
        )
        resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
        resid_fig.update_layout(title_font_size=20, title_x=0.5)
        plots.append(("Residuals vs Predicted", resid_fig))
    except Exception as e:
        logger.error(f"Error rendering residual plot: {e}")
        st.error(f"Error rendering residual plot: {e}")
    
    # 6. Feature Importance
    try:
        importance = model.named_steps['xgb'].feature_importances_
        features = X_test.columns
        feat_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=False)
        feat_fig = px.bar(
            feat_df,
            x='Importance',
            y='Feature',
            title="Feature Importance",
            template='plotly_white',
            color_discrete_sequence=['#00695c']
        )
        feat_fig.update_layout(title_font_size=20, title_x=0.5)
        plots.append(("Feature Importance", feat_fig))
    except Exception as e:
        logger.error(f"Error rendering feature importance plot: {e}")
        st.error(f"Error rendering feature importance plot: {e}")
    
    return plots

# Main Streamlit app
def main():
    load_fonts()
    load_css()
    
    st.markdown('<div class="title">Concrete Strength Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict Compressive Strength with Fly Ash or RHA using XGBoost</div>', unsafe_allow_html=True)
    
    # How to Use Section
    with st.expander("📖 How to Use This App", expanded=False):
        st.markdown("""
            <div class="how-to-use">
                <h3>Welcome to the Concrete Strength Predictor!</h3>
                
                This app helps you predict the compressive strength of concrete using Fly Ash or Rice Husk Ash (RHA) with a powerful XGBoost model. Follow these steps to get started:
                
                <hr>
                
                - **Choose an Ash Type**:  
                  In the sidebar, select **Flyash** or **RHA** to specify the material for your concrete mix.
                
                - **Predict Without a Dataset**:  
                  - Navigate to the **Strength Prediction** section below.  
                  - Enter values for Water, Cement, Fine Aggregate, Coarse Aggregate, Fly Ash/RHA, Age, and Super Plasticizer (in kg/m³ or days).  
                  - Click **Predict Strength** to view the predicted compressive strength in MPa.
                
                - **Train with Your Dataset**:  
                  - In the sidebar, upload a CSV file under **Upload Flyash/RHA CSV**.  
                  - Your CSV must include these columns (all numeric, no missing values):  
                    - Water (kg/m3)  
                    - Cement (kg/m3)  
                    - Fine aggregate (kg/m3)  
                    - Coarse aggregate (kg/m3)  
                    - Fly ash (kg/m3) *(for Flyash)* or Rice husk ash (kg/m3) *(for RHA)*  
                    - Age (days)  
                    - Super plasticizer (kg/m3)  
                    - Compressive strength (MPa)  
                  - The app will train a new model, save it, and display:  
                    - **Model Performance**: Metrics like MSE, MAE, R², and Cross-Validation RMSE.  
                    - **Data Insights**: Interactive plots (Scatter, Pair, Heatmap, Distributions, Residuals, Feature Importance).  
                  - Use the **Strength Prediction** section to predict with the new model.
                
                - **Tips for Success**:  
                  - Use realistic values (e.g., Water: 150-200 kg/m³, Age: 7-90 days).  
                  - Ensure your CSV has at least 10 rows and matches the selected ash type.  
                  - Check error messages if predictions or uploads fail, and verify your inputs or dataset.
                
                <hr>
                
                Enjoy predicting concrete strength with ease! 🚀
            </div>
        """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        ash_type = st.selectbox(
            "Alternative Material", ["Flyash", "RHA"],
            help="Choose Fly Ash or Rice Husk Ash (RHA) for prediction"
        )
        st.header("Dataset")
        uploaded_file = st.file_uploader(
            f"Upload {ash_type} CSV", type=['csv'],
            help="Upload a CSV with required columns to train a new model"
        )
        st.markdown('<div class="credit-text">Made by Ayush Sharma</div>', unsafe_allow_html=True)
    
    # Load saved model for predictions
    with st.spinner(f"Loading {ash_type} XGBoost model..."):
        model = load_model(ash_type)
    
    # Handle uploaded dataset
    file_path = None
    data = None
    X, y = None, None
    mse, mae, r2, cv_mean_rmse, cv_std_rmse, best_params = None, None, None, None, None, None
    X_test, y_test, y_pred = None, None, None
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name
        
        # Load data
        with st.spinner("Loading and validating dataset..."):
            X, y, data = load_data(file_path, ash_type)
        
        if X is None or y is None or data is None:
            try:
                os.remove(file_path)
            except:
                pass
            return
        
        # Train model
        with st.spinner("Training XGBoost model... This may take a moment."):
            model, mse, mae, r2, X_test, y_test, y_pred, best_params, cv_mean_rmse, cv_std_rmse = train_model(X, y)
        
        if model is None:
            try:
                os.remove(file_path)
            except:
                pass
            return
        
        # Save trained model
        save_model(model, ash_type)
        
        # Display metrics
        st.header("Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Mean Squared Error</h3>
                    <p>{mse:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Mean Absolute Error</h3>
                    <p>{mae:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>R² Score</h3>
                    <p>{r2:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Cross-Val RMSE</h3>
                    <p>{cv_mean_rmse:.2f} ± {cv_std_rmse:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<span class="hyperparam-text">**Best Hyperparameters:**</span>', unsafe_allow_html=True)
        st.json(best_params)
        
        # Dataset visualization
        st.header("Data Insights")
        try:
            plots = generate_plots(data, X_test, y_test, y_pred, model, ash_type)
            for title, plot in plots:
                st.subheader(title)
                st.plotly_chart(plot, use_container_width=True)
        except Exception as e:
            logger.error(f"Error rendering plots: {e}")
            st.error(f"Error rendering plots: {e}")
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
    
    # Prediction section
    st.header("Strength Prediction")
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            water = st.number_input(
                "Water (kg/m³)", min_value=0.0, value=180.0, step=5.0,
                help="Enter water quantity in kg/m³"
            )
            cement = st.number_input(
                "Cement (kg/m³)", min_value=0.0, value=350.0, step=10.0,
                help="Enter cement quantity in kg/m³"
            )
            fine_agg = st.number_input(
                "Fine Aggregate (kg/m³)", min_value=0.0, value=700.0, step=10.0,
                help="Enter fine aggregate quantity in kg/m³"
            )
            coarse_agg = st.number_input(
                "Coarse Aggregate (kg/m³)", min_value=0.0, value=1100.0, step=10.0,
                help="Enter coarse aggregate quantity in kg/m³"
            )
        
        with col2:
            ash_label = "Fly ash (kg/m³)" if ash_type == "Flyash" else "Rice husk ash (kg/m³)"
            ash_value = 100.0 if ash_type == "Flyash" else 50.0
            ash = st.number_input(
                ash_label, min_value=0.0, value=ash_value, step=5.0,
                help=f"Enter {'Fly ash' if ash_type == 'Flyash' else 'RHA'} quantity in kg/m³"
            )
            age = st.number_input(
                "Age (days)", min_value=0.0, value=28.0, step=1.0,
                help="Enter curing age in days"
            )
            super_plasticizer = st.number_input(
                "Super Plasticizer (kg/m³)", min_value=0.0, value=5.0, step=0.5,
                help="Enter super plasticizer quantity in kg/m³"
            )
        
        if st.button("Predict Strength", use_container_width=True):
            if model is None:
                st.error(f"Cannot predict: No {ash_type} model available. Please ensure '{ash_type.lower()}_model.pkl' exists in the app directory or upload a dataset to train a new model.")
            else:
                with st.spinner("Predicting compressive strength..."):
                    new_mix = [water, cement, fine_agg, coarse_agg, ash, age, super_plasticizer]
                    predicted_strength = predict_strength(model, new_mix, ash_type)
                    if predicted_strength is not None:
                        st.markdown(f"<div class='prediction-result'>Predicted Compressive Strength: {predicted_strength:.2f} MPa</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()