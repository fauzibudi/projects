import streamlit as st
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from datetime import timedelta
import logging
import random

# Seed untuk konsistensi
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Harga Bitcoin dengan LSTM", layout="centered")
st.title("Prediksi Harga Bitcoin dengan LSTM")

# Input user
time_step = st.number_input("Time Step (min_value=10, max_value=100)", min_value=10, max_value=100, value=50)
epochs = st.number_input("Number of EPOCH for training (min_value=10, max_value=100)", min_value=10, max_value=100, value=50)

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
if 'training_done' not in st.session_state:
    st.session_state.training_done = False

# Variabel default
days = 365
future_days = 30

# Inisialisasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Fungsi untuk mengambil data dari CoinGecko
def get_crypto_data(days):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if 'prices' not in data:
            logging.error("Data harga tidak ditemukan.")
            st.error("Gagal mengambil data harga. Periksa koneksi.")
            return None
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([p[0] for p in data['prices']], unit='ms'),
            'price': [p[1] for p in data['prices']]
        })
        df.set_index('timestamp', inplace=True)
        df = df.dropna()
        if df.empty:
            logging.error("Data kosong setelah penghapusan NaN.")
            st.error("Data kosong. Periksa parameter.")
            return None
        return df
    except requests.RequestException as e:
        logging.error(f"Error fetching price data: {str(e)}")
        st.error("Gagal mengambil data. Periksa koneksi internet.")
        return None

# Fungsi untuk membuat dataset
def create_dataset(data, time_step):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Fungsi untuk membangun dan melatih model
def train_model(X_train, Y_train, epochs):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=32, callbacks=[early_stopping], verbose=0)
    return model

# Tampilkan grafik Bitcoin secara default
if st.session_state.data is None:
    st.write("Mengambil data historis Bitcoin untuk ditampilkan...")
    try:
        price_data = get_crypto_data(days)
        if price_data is not None:
            st.session_state.data = price_data['price'].sort_index(ascending=True)[-365:]
            st.subheader("Grafik Harga Bitcoin Historis (365 Hari Terakhir)")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(st.session_state.data.index, st.session_state.data.values, label='Harga Historis', color='blue', linewidth=2)
            ax.set_title('Harga Bitcoin Historis (USD)', fontsize=14, pad=10)
            ax.set_xlabel('Tanggal', fontsize=12)
            ax.set_ylabel('Harga (USD)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data historis: {str(e)}")

# Tombol untuk mengambil data
if st.button("Ambil Data Harga"):
    st.write("Mengambil data harga Bitcoin...")
    try:
        price_data = get_crypto_data(days)
        if price_data is None:
            st.stop()
        st.session_state.data = price_data['price'].sort_index(ascending=True)[-365:]
        st.success("Data harga berhasil diambil!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data: {str(e)}")

# Tombol untuk proses training dan evaluasi
if st.session_state.data is not None and st.button("Proses Training dan Evaluasi"):
    st.write("Preprocessing data...")
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(st.session_state.data.values.reshape(-1, 1))
        st.session_state.scaler = scaler

        X, Y = create_dataset(scaled_data, time_step)
        if len(X) == 0 or len(Y) == 0:
            st.error("Data tidak cukup untuk membuat sequence.")
            st.stop()

        train_size = int(len(X) * 0.80)
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        Y_train = Y_train.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)

        st.write("Melatih model LSTM...")
        with st.spinner("Ini mungkin memakan waktu beberapa menit..."):
            model = train_model(X_train, Y_train, epochs)
        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.Y_test = Y_test
        st.success("Model selesai dilatih!")

        st.write("Mengevaluasi model...")
        test_predict = st.session_state.model.predict(X_test, verbose=0)
        test_predict = test_predict.reshape(-1, 1)
        Y_test_inv = st.session_state.scaler.inverse_transform(Y_test)
        test_predict_inv = st.session_state.scaler.inverse_transform(test_predict)
        rmse = math.sqrt(mean_squared_error(Y_test_inv, test_predict_inv))
        mae = mean_absolute_error(Y_test_inv, test_predict_inv)

        st.subheader("Hasil Evaluasi Model")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"{rmse:.2f} USD")
        with col2:
            st.metric("MAE", f"{mae:.2f} USD")

        st.subheader("Grafik Prediksi pada Data Uji")
        fig, ax = plt.subplots(figsize=(10, 5))
        dates = st.session_state.data.index[-len(Y_test_inv):]
        ax.plot(dates, Y_test_inv, label='Aktual', color='blue', linewidth=2)
        ax.plot(dates, test_predict_inv, label='Prediksi', color='orange', linewidth=2, linestyle='--')
        ax.set_title('Aktual vs Prediksi Harga Bitcoin', fontsize=14, pad=10)
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Harga (USD)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.session_state.training_done = True
    except Exception as e:
        st.error(f"Terjadi kesalahan saat training atau evaluasi: {str(e)}")

# Tombol untuk prediksi 30 hari ke depan
if st.session_state.training_done and st.button("Prediksi 30 Hari ke Depan"):
    st.write("Membuat prediksi untuk 30 hari ke depan...")
    try:
        last_sequence = st.session_state.scaler.transform(st.session_state.data.values[-time_step:].reshape(-1, 1))
        last_sequence = last_sequence.reshape((1, time_step, 1))
        predictions = []
        current_input = last_sequence.copy()
        for _ in range(future_days):
            next_pred = st.session_state.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, 0] = next_pred

        predictions_inv = st.session_state.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = [st.session_state.data.index[-1] + timedelta(days=i+1) for i in range(future_days)]

        st.subheader("Prediksi Harga Bitcoin 30 Hari ke Depan")
        fig, ax = plt.subplots(figsize=(10, 5))
        historical_dates = st.session_state.data.index[-252:]
        ax.plot(historical_dates, st.session_state.data.values[-252:], label='Historis', color='blue', linewidth=2)
        ax.plot(future_dates, predictions_inv, label='Prediksi', color='orange', linewidth=2, linestyle='--')
        ax.set_title('Historis vs Prediksi Harga Bitcoin (USD)', fontsize=14, pad=10)
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Harga (USD)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.info(f"Data diambil untuk {days} hari terakhir. Prediksi dibuat untuk {future_days} hari ke depan. Model dilatih dengan time_step={time_step} dan {epochs} epoch.")
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam prediksi: {str(e)}")

# Catatan
st.write("**Catatan**:")
st.write("- Data diambil dari CoinGecko untuk 365 hari terakhir.")
st.write("- Model dilatih dengan time_step dan epochs yang ditentukan pengguna.")
st.write("- Bukan saran keuangan!")