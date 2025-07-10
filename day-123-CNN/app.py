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

# --- DEBUGGING SECTION ---
st.subheader("Informasi Debugging (Hanya untuk Pengembangan)")
st.write(f"Direktori Kerja Saat Ini: `{os.getcwd()}`")
st.write("Daftar File di Direktori Kerja:")
try:
    files_in_cwd = os.listdir(os.getcwd())
    for f in files_in_cwd:
        st.write(f"- `{f}`")
except Exception as e:
    st.error(f"Gagal membaca direktori: {e}")
st.write("---")
# --- END DEBUGGING SECTION ---


# --- Load Model ---
@st.cache_resource # Cache model agar tidak dimuat ulang setiap kali interaksi
def load_model():
    """
    Memuat model Keras yang telah dilatih.
    Asumsi: Model berada di folder yang sama dengan app.py
    """
    model_path = 'best_butterfly_classification_model_transfer_learning.h5'
    if not os.path.exists(model_path):
        st.error(f"Error: File model '{model_path}' tidak ditemukan. Pastikan model berada di folder yang sama dengan app.py.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# --- Load Class Names ---
@st.cache_data # Cache data nama kelas
def load_class_names():
    """
    Memuat nama kelas dari file class_indices.json yang disimpan saat pelatihan.
    """
    class_indices_path = 'class_indices.json' # Asumsi file ini ada di folder yang sama
    if not os.path.exists(class_indices_path):
        st.error(f"Error: File '{class_indices_path}' tidak ditemukan. Harap pastikan Anda telah menyimpan class_indices.json saat pelatihan.")
        st.info("Anda perlu menambahkan kode untuk menyimpan `train_generator.class_indices` ke `class_indices.json` di skrip pelatihan Anda dan menjalankan ulang pelatihan.")
        return None
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        # Rekonstruksi daftar CLASS_NAMES berdasarkan indeks integer
        # Ini memastikan urutan yang benar, karena class_indices adalah dict {nama: indeks}
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
    # Periksa apakah jumlah kelas yang dimuat cocok dengan output model
    if len(CLASS_NAMES) != model.output_shape[1]:
        st.warning(f"Jumlah kelas yang dimuat ({len(CLASS_NAMES)}) tidak cocok dengan output model ({model.output_shape[1]}). Prediksi mungkin tidak akurat. Pastikan `class_indices.json` sesuai dengan model yang dilatih.")
        display_class_names = False # Set flag untuk tidak menampilkan nama kelas
    else:
        display_class_names = True

    # --- Fungsi untuk Preprocessing Gambar ---
    def preprocess_image(img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
        img_array = img_array / 255.0 # Normalisasi piksel
        return img_array

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Pilih gambar kupu-kupu...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        st.image(uploaded_file, caption='Gambar yang Diunggah.', use_column_width=True)
        st.write("")
        st.write("Memprediksi...")

        # Preprocess gambar
        processed_image = preprocess_image(uploaded_file, target_size=(224, 224))

        # Lakukan prediksi
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        # Tampilkan hasil prediksi
        if display_class_names: # Gunakan flag untuk memutuskan apakah akan menampilkan nama kelas
            predicted_label = CLASS_NAMES[predicted_class_index]
            st.success(f"Ini adalah **{predicted_label}** dengan keyakinan **{confidence:.2f}%**.")
        else:
            st.warning("Tidak dapat menampilkan nama kelas karena daftar kelas tidak cocok dengan model.")
            st.success(f"Kelas yang diprediksi (indeks): {predicted_class_index} dengan keyakinan {confidence:.2f}%.")

        st.subheader("Probabilitas untuk Semua Kelas:")
        # Buat DataFrame untuk menampilkan probabilitas
        if display_class_names: # Gunakan flag untuk memutuskan apakah akan menampilkan nama kelas
            prob_df = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Probability': predictions[0]
            }).sort_values(by='Probability', ascending=False)
            st.dataframe(prob_df, use_container_width=True)
        else:
            st.dataframe(pd.DataFrame({'Index': range(len(predictions[0])), 'Probability': predictions[0]}), use_container_width=True)

else: # Blok ini menangani kasus di mana model atau CLASS_NAMES tidak dapat dimuat
    if model is None:
        st.warning("Model tidak dapat dimuat. Pastikan file model ada di lokasi yang benar.")
    if CLASS_NAMES is None:
        st.warning("Nama kelas tidak dapat dimuat. Pastikan file `class_indices.json` ada dan benar.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ oleh AI")
