import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from streamlit_folium import st_folium
import folium

st.subheader("Pilih Lokasi di Peta")

# Tampilkan peta interaktif
m = folium.Map(location=[-7.9797, 112.6304], zoom_start=5)
marker = folium.Marker(location=[-7.9797, 112.6304], draggable=True)

map_data = st_folium(m, width=700, height=500)

# Ambil koordinat dari klik pengguna
if map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"Lokasi terpilih: Latitude={lat:.4f}, Longitude={lon:.4f}")
else:
    st.warning("Klik lokasi di peta untuk memilih koordinat.")
    st.stop()

# Fungsi untuk mengambil data cuaca dari Open-Meteo
@st.cache_data
def fetch_weather_data(lat, lon):
    start_date = datetime.today().date() - timedelta(days=60)
    end_date = datetime.today().date()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloudcover,weathercode",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()["hourly"]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.hour
        df["day"] = df["time"].dt.dayofweek
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengambil data: {e}")
        return None

# Fungsi untuk melatih model
@st.cache_resource
def train_models(df):
    # Split data untuk regresi
    X_base = df[["hour", "day"]]
    y_targets = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloudcover"]
    X_train, X_test, y_train, y_test = train_test_split(X_base, df[y_targets], test_size=0.2, random_state=42)
    
    # Latih model regresi
    regressors = {}
    for target in y_targets:
        model = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train[target])
        regressors[target] = model
    
    # Latih model klasifikasi curah hujan
    bins = [-0.1, 0.1, 2, 100]
    labels = [0, 1, 2]
    df["precip_class"] = pd.cut(df["precipitation"], bins=bins, labels=labels).astype(int)
    X_p = df[["hour", "day"]]
    y_p = df["precip_class"]
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
    precip_model = SVC(C=0.1, class_weight='balanced', gamma='auto', kernel='rbf', random_state=42)
    precip_model.fit(X_train_p, y_train_p)
    
    # Latih model klasifikasi weathercode
    X_w = df[["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "cloudcover", "hour", "day"]]
    y_w = df["weathercode"]
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
    weather_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    weather_model.fit(X_train_w, y_train_w)
    
    return regressors, precip_model, weather_model

# Fungsi untuk membuat DataFrame masa depan
def generate_future_df():
    future = pd.date_range(start=datetime.now(), periods=168, freq='h')
    future_df = pd.DataFrame({'time': future})
    future_df['hour'] = future_df['time'].dt.hour
    future_df['day'] = future_df['time'].dt.dayofweek
    return future_df

# Fungsi untuk memetakan weathercode ke deskripsi
def map_weathercode(code):
    if   code == 0:
        return "Cerah 🌞"
    elif code in [1, 2, 3]:
        return "Berawan ☁️"
    elif code in [45, 48]:
        return "Berkabut"
    elif code in [51, 53, 55, 56, 57]:  
        return "Gerimis 💧"
    elif code in [61, 63, 65, 80]:
        return "Hujan Ringan 🌦️"
    elif code in [66, 67, 81, 82]:
        return "Hujan Deras 🌧️"
    elif code in [71, 73, 75, 85, 86, 77]:
        return "Bersalju ❄️"
    elif code in [95, 96, 99]:
        return "Hujan Petir ⛈️"
    else:
        return "Lainnya"

# Antarmuka Streamlit
st.title("Prediksi Cuaca 7 Hari ke Depan")
st.write("Pilih lokasi berdasarkan peta di atas atau masukkan koordinat Latitude dan Longitude untuk lokasi yang diinginkan (contoh: Malang, -7.9797, 112.6304).")

# Input pengguna
lat = st.number_input("Lintang (Latitude)", value=lat, format="%.4f")
lon = st.number_input("Bujur (Longitude)", value=lon, format="%.4f")

if st.button("Confirm"):
    with st.spinner("Mengambil data dan melatih model..."):
        # Ambil dan proses data
        df = fetch_weather_data(lat, lon)
        if df is None:
            st.stop()
        
        # Latih model
        regressors, precip_model, weather_model = train_models(df)
        
        # Buat DataFrame masa depan
        future_df = generate_future_df()
        
        # Prediksi fitur kontinu
        for target, model in regressors.items():
            future_df[target] = model.predict(future_df[['hour', 'day']])
        
        # Prediksi curah hujan
        future_df['precip_class'] = precip_model.predict(future_df[['hour', 'day']])
        future_df['precipitation'] = future_df['precip_class'].apply(lambda x: {0: 0.0, 1: 1.0, 2: 5.0}[x])
        
        # Prediksi weathercode
        features_for_weather = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "cloudcover", "hour", "day"]
        future_df['weathercode_pred'] = weather_model.predict(future_df[features_for_weather])
        future_df['weather_pred'] = future_df['weathercode_pred'].apply(map_weathercode)
        
        # Tampilkan hasil dalam tabel
        st.subheader("Prediksi Cuaca 7 Hari ke Depan")
        display_df = future_df[['time', 'temperature_2m', 'precipitation', 'relative_humidity_2m', 'wind_speed_10m', 'weather_pred']].copy()
        display_df['temperature_2m'] = display_df['temperature_2m'].round(1)
        display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H.00')
        display_df.rename(columns={
            'temperature_2m': 'Suhu (°C)',
            'precipitation': 'Curah Hujan (mm)',
            'weather_pred': 'Kondisi Cuaca',
            'relative_humidity_2m': 'Kelembapan (%)',
            'wind_speed_10m': 'Kecepatan Angin (km/jam)'
        }, inplace=True)
        st.dataframe(display_df)