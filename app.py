import streamlit as st
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import hopsworks
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta, date

def prepare_data_for_forecast(df):
    sorted_df = df.sort_values(by='id')

    high_prices = sorted_df.loc[:, 'high'].values
    low_prices = sorted_df.loc[:, 'low'].values
    mid_prices = (high_prices + low_prices) / 2.0

    mid_price_changes = np.diff(mid_prices) / mid_prices[:-1] * 100
    mid_price_changes = np.insert(mid_price_changes, 0, 0)

    features = sorted_df[
        ['volume', 'ma7', 'ma21', 'bollinger_upper', 'bollinger_lower', 'volatility',
         'close_usd_index', 'close_oil', 'close_gold', 'hash_rate']].values
    feature_changes = np.diff(features, axis=0) / features[:-1] * 100
    feature_changes = np.insert(feature_changes, 0, 0, axis=0)

    combined_features = np.column_stack((mid_price_changes.reshape(-1, 1), feature_changes))

    sequence_length = 100
    sequence_data = []
    sequence_labels = []

    for i in range(len(combined_features) - sequence_length):
        sequence_data.append(combined_features[i:i + sequence_length])
        # Labels based on whether the next mid_price_change is positive (1) or negative (0)
        sequence_labels.append(1 if mid_price_changes[i + sequence_length] > 0 else 0)

    sequence_data = np.array(sequence_data)
    sequence_labels = np.array(sequence_labels)


    split_index = int(len(sequence_data) * 0.8)
    train_data = sequence_data[:split_index]
    train_labels = sequence_labels[:split_index]
    test_data = sequence_data[split_index:]
    test_labels = sequence_labels[split_index:]

    train_data = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    input_size = combined_features.shape[1]

    all_data = np.concatenate(
        (train_data.tensors[0].numpy(), test_data.tensors[0].numpy())).reshape(-1, input_size)

    return all_data


def forecast(model, data, sequence_length, steps_ahead):
    model.eval()

    data = np.array(data)

    current_sequence = data[-sequence_length:].reshape(1, sequence_length, -1)
    predictions = []

    with torch.no_grad():
        for _ in range(steps_ahead):
            input_seq = torch.tensor(current_sequence, dtype=torch.float32)

            output = model(input_seq)
            predicted_value = torch.sigmoid(output).item()

            predicted_class = 1 if predicted_value >= 0.5 else 0

            predictions.append(predicted_class)

            new_sequence = np.append(current_sequence[0, 1:, :],
                                     [[predicted_class] * current_sequence.shape[2]], axis=0)
            current_sequence = new_sequence.reshape(1, sequence_length, -1)

            movement_interpretation = ['Increase' if pred == 1 else 'Decrease' for pred in predictions]

    return movement_interpretation

@st.cache_data
def load_hopsworks_data():
    """
    Load data from Hopsworks and cache the result.
    """
    project = hopsworks.login()
    fs = project.get_feature_store()
    bitcoin_fg = fs.get_feature_group(name='bitcoin_price_movement', version=2)
    data = bitcoin_fg.select_all()
    feature_view = fs.get_or_create_feature_view(
        name='bitcoin_price_movement_training_fv',
        version=2,
        query=data
    )
    df = feature_view.get_batch_data()
    sorted_df = prepare_data_for_forecast(df)
    return sorted_df

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
        ax3.spines["right"].set_position(("outward", 60))
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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out
    
def load_model():
    # import the model
    project = hopsworks.login()
    mr = project.get_model_registry()
    model = mr.get_model(
    name="bitcoin_price_movement_prediction_model_lstm", 
    version=1
    )

    saved_model_dir = model.download()

    lstm_model = LSTMModel(input_size=11, hidden_size=30, output_size=1)
    lstm_model.load_state_dict(torch.load(saved_model_dir + "/bitcoin_price_movement_prediction_lstm.pth"))
    lstm_model.eval()
    return lstm_model

def get_today_bitcoin_price():
    btc_price = yf.download('BTC-USD', period='1d')['Close'].values[0]
    return btc_price

def get_bitcoin_price_selected_dates(start_date, end_date):
    return yf.download('BTC-USD', start=start_date, end=end_date)['Close'].values[0]
    
def main():
    #set_background('images/background.jpg')

    st.title('Bitcoin Price Movement Forecast')

    st.write("""
        Welcome to our Bitcoin Trend Predictor! This app is designed to help you monitor and evaluate the performance of 
        your Bitcoin investments with ease. Whether you're a trader, investor, or cryptocurrency enthusiast dive into 
        our interactive app and make informed decisions in the fast-paced world of cryptocurrency. 
    """)

    lstm_model = load_model()

    # Load data from Hopsworks
    sorted_df= load_hopsworks_data()
    
    movement_interpretation = forecast(lstm_model, sorted_df, 100, 2)

    st.header('Bitcoin portfolio calculator')

    # Create a calendar selector
    selected_date = st.date_input("Select a date", value=None, min_value=datetime(2014, 10, 1), max_value=date.today())
    if selected_date:
        selected_date = selected_date
        following_day_date = selected_date + timedelta(days=1)
        btc_price_selected_date = get_bitcoin_price_selected_dates(selected_date.strftime('%Y-%m-%d'), following_day_date.strftime('%Y-%m-%d'))
        btc_price_today = get_today_bitcoin_price()

        # Display the selected date
        st.write("The price of the bitcoin at your selected date is: ", btc_price_selected_date)
        st.write("The price of the bitcoin today is: ", btc_price_today)

        btc_owned = st.number_input('Enter the amount of Bitcoin you own', min_value=0.0, step=0.01)

        if btc_owned > 0:
            # Calculate the value of the investment on the selected date and today
            initial_investment_value = btc_owned * btc_price_selected_date
            current_investment_value = btc_owned * btc_price_today

            # Calculate the gain or loss
            gain_loss = current_investment_value - initial_investment_value
            gain_loss_percentage = (gain_loss / initial_investment_value) * 100

            # Display the investment results
            st.markdown(
                f"You invested **\${initial_investment_value}** on {selected_date.strftime('%Y-%m-%d')}. "
                f"Today, your investment is worth **${current_investment_value}**."
            )

            if gain_loss > 0:
                st.write(f"You have a gain of ${gain_loss:.2f} ({gain_loss_percentage:.2f}%)")
            else:
                st.write(f"You have a loss of ${-gain_loss:.2f} ({gain_loss_percentage:.2f}%)")

            if movement_interpretation:
                st.header('Bitcoin Price Prediction for the Next 2 Days')

                day_1_prediction = movement_interpretation[0]
                day_2_prediction = movement_interpretation[1]

                day_1_emoji = "📈" if day_1_prediction == "Increase" else "📉"
                day_2_emoji = "📈" if day_2_prediction == "Increase" else "📉"

                st.write(f"Day 1 Prediction: {day_1_prediction.capitalize()} {day_1_emoji}")
                st.write(f"Day 2 Prediction: {day_2_prediction.capitalize()} {day_2_emoji}")

                # Suggestion based on prediction and current gain/loss
                if movement_interpretation[0] == "Increase" or movement_interpretation[1] == "Increase":
                    if gain_loss > 0:
                        suggestion = "sell"
                    else:
                        suggestion = "hold"
                elif movement_interpretation[0] == "Decrease" and movement_interpretation[1] == "Decrease":
                    suggestion = "buy"
                else:
                    suggestion = "hold"

                st.subheader(f"Suggestion: It may be a good time to {suggestion}.")
            else:
                st.write("Prediction data is not available.")

    # plot visualization
    sorted_df = load_hopsworks_data()
    st.header('Visualizations')
    visualization_option = st.selectbox(
        'Choose a visualization:',
        ('Select an option',
         'Bitcoin Price and USD Index',
         'Bitcoin Price, Oil Price, and Gold Price',
         'Boxplot of Bitcoin Closing Prices by Year')
    )

    if visualization_option != 'Select an option':
        plot_visualization(visualization_option, sorted_df)




if __name__ == "__main__":
    main()