import streamlit as st
import pandas as pd
import hopsworks 
import joblib
import os
from dotenv import load_dotenv
from features import feature_engineering

def print_fancy_header(text, font_width="bold", font_size=22, color="#2656a3"):
    res = f'<span style="font-width:{font_width}; color:{color}; font-size:{font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)  

def print_fancy_subheader(text, font_width="bold", font_size=22, color="#333"):
    res = f'<span style="font-width:{font_width}; color:{color}; font-size:{font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)  

@st.cache_data 
def load_data():

    project = hopsworks.login()

    fs = project.get_feature_store()
    mr = project.get_model_registry()

    feature_view = fs.get_feature_view(
    name='bitcoin_training_fv',
    version=1
    )

    model = mr.get_model(
    name="bitcoin_price_prediction_model", 
    version=1
    )

    saved_model_dir = model.download()

    xgboost_model = joblib.load(saved_model_dir + "/bitcoin_price_prediction_model.pkl")

    bitcoin_fg = fs.get_feature_group(
    name='bitcoin_price',
    version=2,
    )

    data = bitcoin_fg.select_all()
    version = 1 
    feature_view = fs.get_or_create_feature_view(
    name='bitcoin_training_fv',
    version=version,
    query=data
    )

    df = feature_view.get_batch_data()
    sorted_df = df.sort_values(by='timestamp')
    

    st.write("New data:")
    last_row=sorted_df.tail(1)
    st.dataframe(last_row)            

    X = sorted_df.drop(columns=['timestamp', 'date', 'tomorrow'])

    day_to_predict = X.tail(1).iloc[0]
    day_to_predict_reshaped = day_to_predict.values.reshape(1, -1)
    pred = xgboost_model.predict(day_to_predict_reshaped)

    prediction_df = pd.DataFrame({'Date of last available closing price': sorted_df['date'].iloc[-1], 'Predicted value for the next day': pred})

    return prediction_df

#########################

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)

st.title('Bitcoin Price Prediction')

st.markdown("""
            Trying to predict tomorrow's closing Bitcoin price using XGBoost Regressor model.
""")

st.write(3 * "-")

with st.expander("📊 **Data Engineering and Machine Learning Operations in Business**"):
                 st.markdown("""
LEARNING OBJECTIVES
- Using our skills for designing, implementing, and managing data pipelines and ML systems.
- Focus on practical applications within a business context.
- Cover topics such as data ingestion, preprocessing, model deployment, monitoring, and maintenance.
- Emphasize industry best practices for effective operation of ML systems.
"""
)
                 
with st.expander("📊 **This assigment**"):
                 st.markdown("""
The objective of this assignment is to build a prediction system that predicts the price of Bitcoin using historical prices and time-series analysis.
"""
)
                 
with st.sidebar:

    print_fancy_header('\n📡 Connecting to Hopsworks Feature Store...')

    st.write("Logging... ")
    load_dotenv()
    api_key = os.getenv('HOPSWORKS_FS_API_KEY')
    project = hopsworks.login(project = "mlopsbds", api_key_value=api_key)
    fs = project.get_feature_store()
    progress_bar.progress(40)
    st.write("✅ Logged in successfully!")

st.write(3 * "-")
print_fancy_header('\n Retriving batch data from Feature Store...')

predictions_df = load_data()

progress_bar.progress(100)

st.write(predictions_df)