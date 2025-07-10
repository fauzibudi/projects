import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import json

# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    page_title="Klasifikasi Kupu-Kupu Cerdas",
    page_icon="🦋",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🦋 Klasifikasi Kupu-Kupu Cerdas")
st.write("Unggah gambar kupu-kupu untuk diidentifikasi oleh model AI kami.")

# --- Load Model ---
@st.cache_resource 
def load_model():
    """
    Memuat model Keras yang telah dilatih.
    Asumsi: Model berada di folder 'day-123-CNN' di dalam direktori kerja Streamlit.
    """
    model_path = 'day-123-CNN/best_butterfly_classification_model_transfer_learning.h5'
    if not os.path.exists(model_path):
        st.error(f"Error: File model '{model_path}' tidak ditemukan. Pastikan model berada di folder yang benar.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# --- Load Class Names ---
@st.cache_data 
def load_class_names():
    """
    Memuat nama kelas dari file class_indices.json yang disimpan saat pelatihan.
    Asumsi: File ini berada di folder 'day-123-CNN' di dalam direktori kerja Streamlit.
    """
    class_indices_path = 'day-123-CNN/class_indices.json'
    if not os.path.exists(class_indices_path):
        st.error(f"Error: File '{class_indices_path}' tidak ditemukan. Harap pastikan Anda telah menyimpan class_indices.json saat pelatihan.")
        st.info("Anda perlu menambahkan kode untuk menyimpan `train_generator.class_indices` ke `class_indices.json` di skrip pelatihan Anda dan menjalankan ulang pelatihan.")
        return None
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        sorted_class_names = [None] * len(class_indices)
        for name, index in class_indices.items():
            sorted_class_names[index] = name
        return sorted_class_names
    except Exception as e:
        st.error(f"Gagal memuat nama kelas dari '{class_indices_path}': {e}")
        return None

CLASS_NAMES = load_class_names()

# --- Jika model dan nama kelas berhasil dimuat, lanjutkan ---
if model is not None and CLASS_NAMES is not None:
    if len(CLASS_NAMES) != model.output_shape[1]:
        st.warning(f"Jumlah kelas yang dimuat ({len(CLASS_NAMES)}) tidak cocok dengan output model ({model.output_shape[1]}). Prediksi mungkin tidak akurat. Pastikan `class_indices.json` sesuai dengan model yang dilatih.")
        display_class_names = False 
    else:
        display_class_names = True

    # --- Fungsi untuk Preprocessing Gambar ---
    def preprocess_image(img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0 
        return img_array

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Pilih gambar kupu-kupu...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Gambar yang Diunggah.', use_container_width=True) 
        st.write("")
        st.write("Memprediksi...")

        processed_image = preprocess_image(uploaded_file, target_size=(224, 224))

        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        if display_class_names: 
            predicted_label = CLASS_NAMES[predicted_class_index]
            st.success(f"Ini adalah **{predicted_label}** dengan keyakinan **{confidence:.2f}%**.")
        else:
            st.warning("Tidak dapat menampilkan nama kelas karena daftar kelas tidak cocok dengan model.")
            st.success(f"Kelas yang diprediksi (indeks): {predicted_class_index} dengan keyakinan {confidence:.2f}%.")

        st.subheader("Probabilitas untuk Semua Kelas:")
        if display_class_names: 
            prob_df = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Probability': predictions[0]
            }).sort_values(by='Probability', ascending=False)
            st.dataframe(prob_df, use_container_width=True)
        else:
            st.dataframe(pd.DataFrame({'Index': range(len(predictions[0])), 'Probability': predictions[0]}), use_container_width=True)

else: 
    if model is None:
        st.warning("Model tidak dapat dimuat. Pastikan file model ada di lokasi yang benar.")
    if CLASS_NAMES is None:
        st.warning("Nama kelas tidak dapat dimuat. Pastikan file `class_indices.json` ada dan benar.")

