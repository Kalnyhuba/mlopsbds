import streamlit as st
import pandas as pd
import os
import hopsworks
import matplotlib.pyplot as plt

# Define constants
INPUT_FOLDER = 'prediction_dataframes'

@st.cache_data
def load_csv_data(filename):
    """
    Load data from a CSV file and cache the result to avoid reloading the file on every rerun.
    """
    file_path = os.path.join(INPUT_FOLDER, filename)
    return pd.read_csv(file_path)

@st.cache_data
def load_hopsworks_data():
    """
    Load data from Hopsworks and cache the result.
    """
    project = hopsworks.login()
    fs = project.get_feature_store()
    bitcoin_fg = fs.get_feature_group(name='bitcoin_analysis', version=1)
    data = bitcoin_fg.select_all()
    feature_view = fs.get_or_create_feature_view(
        name='bitcoin_analysis_training_fv', 
        version=1, 
        query=data
    )
    df = feature_view.get_batch_data()
    sorted_df = df.sort_values(by='id')
    sorted_df['date'] = pd.to_datetime(sorted_df['date'])
    sorted_df['year'] = sorted_df['date'].dt.year
    sorted_df['month'] = sorted_df['date'].dt.month
    sorted_df['day'] = sorted_df['date'].dt.day
    sorted_df['day_of_week'] = sorted_df['date'].dt.dayofweek
    
    return sorted_df

def calculate_investment_value(investment_amount, initial_price, final_price):
    """
    Calculate the investment value based on initial and final prices.
    """
    return (investment_amount / initial_price) * final_price

def plot_visualization(option, df):
    if option == 'Bitcoin Price and USD Index':
        fig, ax1 = plt.subplots(figsize=(14, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Bitcoin Price', color=color)
        ax1.plot(df['date'], df['close'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('USD Index', color=color)
        ax2.plot(df['date'], df['close_usd_index'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Bitcoin Price and USD Index Over Time')
        st.pyplot(fig)
    
    elif option == 'Bitcoin Price, Oil Price, and Gold Price':
        fig, ax1 = plt.subplots(figsize=(14, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Bitcoin Price', color=color)
        ax1.plot(df['date'], df['close'], color=color, label='Bitcoin Price')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Oil Price', color=color)
        ax2.plot(df['date'], df['close_oil'], color=color, label='Oil Price')
        ax2.tick_params(axis='y', labelcolor=color)

        ax3 = ax1.twinx()
        color = 'tab:orange'
        ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
        ax3.set_ylabel('Gold Price', color=color)
        ax3.plot(df['date'], df['close_gold'], color=color, label='Gold Price')
        ax3.tick_params(axis='y', labelcolor=color)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        lines = lines_1 + lines_2 + lines_3
        labels = labels_1 + labels_2 + labels_3
        ax1.legend(lines, labels, loc='upper left')

        fig.tight_layout()
        plt.title('Bitcoin Price, Oil Price, and Gold Price Over Time')
        st.pyplot(fig)

    elif option == 'Boxplot of Bitcoin Closing Prices by Year':
        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract the year from the date column
        df['year'] = pd.to_datetime(df['date']).dt.year

        # Prepare the data for the boxplot
        data_by_year = [group['close'].values for name, group in df.groupby('year')]

        # Plot the boxplots for each year's closing prices
        ax.boxplot(data_by_year, labels=df['year'].unique())
        ax.set_title('Boxplot of Bitcoin Closing Prices by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Closing Price (USD)')
        ax.set_xticklabels(df['year'].unique(), rotation=45)
        ax.grid(True)

        st.pyplot(fig)

def main():
    st.title('Bitcoin Price Prediction')

    st.write("""
        Welcome to our Bitcoin Price Prediction application. 
        Here, you can explore various forecasts and visualizations related to Bitcoin prices. 
        Use the investment calculator to predict the future value of your investments based on the provided data.
    """)

    # Load data from Hopsworks
    sorted_df = load_hopsworks_data()

    # Load CSV data
    forecast_df = load_csv_data('forecast_7_days.csv')
    investment_forecast_df = load_csv_data('forecast_30_days.csv')

    # Display 7-day forecast
    st.header('7-Day Forecast')
    st.write(forecast_df)

    # Investment Calculator
    st.write('### Investment Calculator')
    investment_amount = st.number_input(
        'Enter amount in USD to invest:', 
        min_value=0.0, 
        value=1000.0, 
        step=100.0
    )
    initial_price = sorted_df['close'].iloc[-1]
    final_price = investment_forecast_df['predicted_close'].iloc[-1]

    investment_value = calculate_investment_value(investment_amount, initial_price, final_price)
    st.write(f"If you invest ${investment_amount:.2f} in Bitcoin today, it will be worth ${investment_value:.2f} one month from now.")

    # Visualization options
    st.header('Visualizations')
    visualization_option = st.selectbox(
        'Choose a visualization:',
        ('Bitcoin Price and USD Index', 
         'Bitcoin Price, Oil Price, and Gold Price', 
         'Boxplot of Bitcoin Closing Prices by Year')
    )
    plot_visualization(visualization_option, sorted_df)

    # Display 30-day forecast
    st.header('30-Day Investment Forecast')
    st.write(investment_forecast_df)

if __name__ == "__main__":
    main()