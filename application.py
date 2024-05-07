import streamlit as st
import pandas as pd
import hopsworks 
import joblib
import os
from dotenv import load_dotenv
from features import feature_engineering

# Function to style headers
def print_fancy_header(text, font_weight="bold", font_size="22px", color="#FFD700"):
    st.markdown(
        f'<h2 style="font-weight:{font_weight}; color:{color}; font-size:{font_size};">'
        f'{text}</h2>',
        unsafe_allow_html=True
    )

# Function to style subheaders
def print_fancy_subheader(text, font_weight="bold", font_size="18px", color="#FFD700"):
    st.markdown(
        f'<h3 style="font-weight:{font_weight}; color:{color}; font-size:{font_size};">'
        f'{text}</h3>',
        unsafe_allow_html=True
    )

@st.cache_data 
def load_data():
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    feature_view = fs.get_feature_view(name='bitcoin_training_fv', version=1)
    model = mr.get_model(name="bitcoin_price_prediction_model", version=1)
    saved_model_dir = model.download()
    xgboost_model = joblib.load(saved_model_dir + "/bitcoin_price_prediction_model.pkl")

    bitcoin_fg = fs.get_feature_group(name='bitcoin_price', version=2)
    data = bitcoin_fg.select_all()
    version = 1 
    feature_view = fs.get_or_create_feature_view(name='bitcoin_training_fv', version=version, query=data)
    df = feature_view.get_batch_data()
    sorted_df = df.sort_values(by='timestamp')
    
    st.write("New data:")
    last_row = sorted_df.tail(1)
    st.dataframe(last_row)            

    X = sorted_df.drop(columns=['timestamp', 'date', 'tomorrow'])

    day_to_predict = X.tail(1).iloc[0]
    day_to_predict_reshaped = day_to_predict.values.reshape(1, -1)
    pred = xgboost_model.predict(day_to_predict_reshaped)

    prediction_df = pd.DataFrame({'Date of last available closing price': sorted_df['date'].iloc[-1], 'Predicted value for the next day': pred})

    return prediction_df

# Sidebar header
progress_bar = st.sidebar.header('Loading')

# Title
st.title('Bitcoin Price Prediction')
st.markdown('<style>h1 {color: #FFD700;}</style>', unsafe_allow_html=True)

# Description
st.markdown("""
    Trying to predict tomorrow's closing Bitcoin price using XGBoost Regressor model.
""")

# Data Engineering and Machine Learning Operations in Business expander
with st.expander("**Data Engineering and Machine Learning Operations in Business**"):
    st.markdown("""
        LEARNING OBJECTIVES
        - Planning, managing, and executing complex end-to-end data science projects.
        - Identifying possibilities to deploy machine learning models to real-time and client-facing applications.
        - Evaluating data and machine learning projects, structures, and workflows in organizations.
    """)
    st.markdown('<style>div[role="listbox"] div div div:nth-child(1) {color: #FFD700;}</style>', unsafe_allow_html=True)

# Objective expander
with st.expander("**Objective**"):
    st.markdown("""
        The objective of this assignment is to build a prediction system that predicts the price of Bitcoin using historical prices and time-series analysis.
    """)
    st.markdown('<style>div[role="listbox"] div div div:nth-child(1) {color: #FFD700;}</style>', unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.image('images/bitcoin.jpg', use_column_width=True)
    print_fancy_header('Connecting to Hopsworks Feature Store and retrieving project')
    st.write("Logging in... ")
    load_dotenv()
    api_key = os.getenv('HOPSWORKS_FS_API_KEY')
    project = hopsworks.login(project="mlopsbds", api_key_value=api_key)
    fs = project.get_feature_store()
    progress_bar.progress(40)
    st.write("Logged in successfully!")
    st.write('All data retrieved!')
    st.markdown('<style>div[role="listbox"] div div div:nth-child(1) {color: #FFD700;}</style>', unsafe_allow_html=True)

# Main content
st.write("---")
print_fancy_header('Retrieving the newest available data from Feature Store...')
st.markdown('<style>h2 {color: #FFD700;}</style>', unsafe_allow_html=True)

# Load data
predictions_df = load_data()

# Progress bar completion
progress_bar.progress(100)

# Display prediction dataframe
st.write(predictions_df)
