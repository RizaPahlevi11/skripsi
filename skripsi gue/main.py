import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Judul aplikasi
st.title("Klasifikasi Level Roasting Biji Kopi")

# Definisikan kelas
class_names = {0: 'Dark Roasted', 1: 'Light Roasted', 2: 'Medium Roasted', 3: 'Unroasted'}

# Navigasi halaman
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Beranda", "Unggah Gambar", "Tentang", "Bantuan"])

# Halaman Beranda
if page == "Beranda":
    st.header("Selamat Datang di Aplikasi Klasifikasi Level Roasting Biji Kopi")
    st.write("""
    Aplikasi ini menggunakan model deep learning untuk mengklasifikasikan level roasting biji kopi 
    menjadi empat kategori: Dark Roasted, Light Roasted, Medium Roasted, dan Unroasted.
    """)
    st.write("### Contoh Gambar")
    st.image(["darkRoasted.jpg", "lightRoasted.jpg", "mediumRoasted.jpg", "unroasted.jpg"], 
             caption=["Dark Roasted", "Light Roasted", "Medium Roasted", "Unroasted"], 
             width=300)

# Halaman Unggah Gambar
elif page == "Unggah Gambar":
    st.header("Unggah Gambar Biji Kopi")
    uploaded_file = st.file_uploader("Upload gambar biji kopi...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Baca dan tampilkan gambar
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar biji kopi yang diupload.', use_column_width=True)

        # Preprocess gambar
        img = image.resize((224, 224))  # Sesuaikan dengan ukuran input model
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Muat model
        model = tf.keras.models.load_model('model/ResNet50V2-roastingKopi.h5')

        # Prediksi
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction)  # Ambil indeks dengan nilai tertinggi
        predicted_class = class_names[predicted_index]  # Map indeks ke label kelas

        # Tampilkan hasil prediksi
        st.write(f"Prediksi: {predicted_class}")

# Halaman Tentang
elif page == "Tentang":
    st.header("Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan untuk membantu pengguna dalam mengklasifikasikan level roasting biji kopi 
    menggunakan model deep learning berbasis ResNet50V2.
    """)
    st.write("### Pengembang")
    st.write("- Nama Pengembang: [Nama Anda]")
    st.write("- Kontak: [Email Anda]")
    st.write("- Repositori: [Link GitHub Anda]")

# Halaman Bantuan
elif page == "Bantuan":
    st.header("Panduan Penggunaan dan FAQ")
    st.write("### Panduan Penggunaan")
    st.write("1. Pilih halaman 'Unggah Gambar' dari sidebar.")
    st.write("2. Unggah gambar biji kopi yang ingin diklasifikasikan.")
    st.write("3. Tunggu hingga aplikasi menampilkan hasil prediksi.")
    st.write("### FAQ")
    st.write("**Q: Apa saja format gambar yang didukung?**")
    st.write("A: Aplikasi mendukung format JPG, JPEG, dan PNG.")
    st.write("**Q: Bagaimana cara kerja aplikasi ini?**")
    st.write("A: Aplikasi menggunakan model deep learning yang telah dilatih untuk mengenali level roasting biji kopi.")
    st.write("**Q: Apa yang harus dilakukan jika prediksi tidak akurat?**")
    st.write("A: Pastikan gambar yang diunggah jelas dan fokus pada biji kopi. Jika masalah berlanjut, hubungi pengembang.")

