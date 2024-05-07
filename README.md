# Bitcoin Price Prediction System for the MLOps module at AAUBS

Business Problem:

Our prediction system might be helpful for cryptocurrency investors or traders on deciding whether to buy or sell Bitcoin based on the predicted price for tomorrow. This prediction might be only suited for short-term traders, who only utilise the small fluctuations in the price and don't think about trading in the long run.

Our group made a prediction system which attempts to predict the price of Bitcoin for the next day. The data of historical prices and trading volume comes from the Yahoo Finance API and in the feature pipeline it refreshes every day with the latest available data. The feature pipeline is set up as a Github Actions workflow, and is scheduled to run every day at 12:00 Danish local time. In our training pipeline we used an XGBoost regressor model to predict the price, since it is robust and quite accurate. All data is stored in feature groups on the Hopsworks Feature Store platform and read into the model and predictions using Feature View. We also made a Streamlit application, which is a UI to demonstrate the newest available data about Bitcoin price and the predicted value for the next day relative to the accessing of the application.


![data-pipeline](https://github.com/Kalnyhuba/mlopsbds/blob/master/images/data_pipeline.png)
