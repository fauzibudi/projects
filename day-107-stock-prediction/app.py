import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import requests
import os
import tensorflow as tf
import random
from datetime import timedelta

# seed untuk konsistensi
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def check_api_key(api_key, symbol="AAPL"):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if "Error Message" in data:
        return False, data["Error Message"]
    elif "Information" in data:
        return False, data["Information"]
    return True, "API key valid"

# Judul
st.title("Forecasting Stock Price with LSTM")

# Input user
api_key = st.text_input("Enter your Alpha Vantage API key, you can get it at https://www.alphavantage.co/support/#")
symbol = st.text_input("Enter the stock symbol (for example: AAPL)")
epochs = st.number_input("Number of EPOCH for training (min_value=1, max_value=100)", min_value=1, max_value=100, value=10)
retrain = st.checkbox("Retrain the model (if not checked, use an exiting model)")

# Inisialisasi session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'Y_test' not in st.session_state:
    st.session_state.Y_test = None

# Path untuk menyimpan model
MODEL_PATH = "lstm_model.h5"

# Tombol untuk cek API key
if st.button("Check API Key"):
    if not api_key:
        st.error("Enter API key")
    else:
        is_valid, message = check_api_key(api_key)
        if is_valid:
            st.success(message)
        else:
            st.error(f"Failed check API key: {message}")

# Tombol untuk memproses
if st.button("Process"):
    if not api_key or not symbol:
        st.error("Enter the API key and stock symbol")
    else:
        st.write("Collecting data...")
        try:
            ts = TimeSeries(key=api_key, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
            
            if "Information" in meta_data and "Thank you for using Alpha Vantage" in meta_data["Information"]:
                st.error(f"Error API: {meta_data['Information']}")
                st.stop()
            
            if data.empty:
                st.error("Failed to retrive data. Check the stock symbol.")
            else:
                df = data['4. close']
                df = df.sort_index(ascending=True)
                df = df[-1260:]  # 5 tahun terakhir
                st.session_state.data = df
                
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df.values.reshape(-1,1))
                st.session_state.scaler = scaler
                
                def create_dataset(data, time_step=50):
                    X, Y = [], []
                    for i in range(len(data)-time_step):
                        X.append(data[i:(i+time_step), 0])
                        Y.append(data[i + time_step, 0])
                    return np.array(X), np.array(Y)
                
                time_step = 50
                X, Y = create_dataset(scaled_data, time_step)
                
                train_size = int(len(X) * 0.65)
                X_train, X_test = X[:train_size], X[train_size:]
                Y_train, Y_test = Y[:train_size], Y[train_size:]
                
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                # Cek apakah model sudah ada dan tidak perlu dilatih ulang
                if os.path.exists(MODEL_PATH) and not retrain:
                    st.write("Loading the exiting model...")
                    model = load_model(MODEL_PATH)
                else:
                    st.write("Create and train new model...")
                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                    model.add(LSTM(50, return_sequences=True))
                    model.add(LSTM(50))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    from tensorflow.keras.callbacks import EarlyStopping
                    early_stopping = EarlyStopping(monitor='loss', patience=5)
                    model.fit(X_train, Y_train, epochs=epochs, batch_size=64, callbacks=[early_stopping], verbose=0)
                    model.save(MODEL_PATH)  # Simpan model
                    
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.Y_test = Y_test
                
                test_predict = model.predict(X_test)
                test_predict = scaler.inverse_transform(test_predict)
                Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1,1))
                
                test_rmse = math.sqrt(mean_squared_error(Y_test_inv, test_predict))
                st.write(f"RMSE Test: {test_rmse}")
                
                fig, ax = plt.subplots()
                dates = st.session_state.data.index[-len(Y_test_inv):]  # Ambil tanggal sesuai panjang data aktual
                ax.plot(dates, Y_test_inv, label='Actual', color='blue', linewidth=2)
                ax.plot(dates, test_predict, label='Forecast', color='orange', linewidth=2, linestyle='--')
                ax.set_title('Actual vs. Forecast Stock Prices', fontsize=14, pad=10)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Price (USD)', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")

# forecasting
if st.session_state.model is not None and st.button("Forecast 30 days ahead"):
    last_sequence = st.session_state.scaler.transform(st.session_state.data.values[-50:].reshape(-1,1))
    last_sequence = last_sequence.reshape(1, 50, 1)
    
    def forecast_future(model, last_sequence, steps, scaler):
        predictions = []
        current_input = last_sequence.copy()
        for _ in range(steps):
            next_pred = model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            next_pred_3d = np.array([[[next_pred]]])  # Bentuk: (1, 1, 1)
            current_input = np.concatenate((current_input[:,1:,:], next_pred_3d), axis=1)
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        return predictions
    
    future_predictions = forecast_future(st.session_state.model, last_sequence, 30, st.session_state.scaler)
    
    fig, ax = plt.subplots()
    historical_dates = st.session_state.data.index[-252:]  # 252 hari terakhir (~1 tahun)
    ax.plot(historical_dates, st.session_state.data.values[-252:], label='Actual', color='blue', linewidth=2)
    future_dates = [st.session_state.data.index[-1] + timedelta(days=i+1) for i in range(30)]
    ax.plot(future_dates, future_predictions, label='Forecast', color='orange', linewidth=2, linestyle='--')
    ax.set_title('Stock Price Forecast (30 Days Ahead)', fontsize=14, pad=10)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


# Catatan
st.write("**Note**:")
st.write("- Free API has a limit of 25 requests per day.")
st.write("- Use valid stock symbols(for example: AAPL).")
st.write("- The model is stored for prediction consistency. Check 'Re -Train Model' to re -practice.")
st.write("- Not Financial Advice!")