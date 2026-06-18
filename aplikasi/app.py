import streamlit as st

st.set_page_config(
    page_title="Prediksi & Forecasting Dashboard",
    page_icon="👋",
)

st.write("# Selamat datang di Dashboard UAS Data Mining! 👋")

st.sidebar.success("Pilih menu di atas.")

st.markdown(
    """
    Aplikasi ini dibangun untuk menggabungkan dua fungsionalitas dari Proyek 1 dan Proyek 2:
    
    ### P1: Klasifikasi Tingkat Keparahan Kecelakaan (US Accidents)
    Model menggunakan algoritma **Random Forest**. Memprediksi tingkat keparahan berdasarkan fitur-fitur kondisi lingkungan, lokasi, dll.
    👉 Buka menu **P1 App** di sidebar.
    
    ### P2: Forecasting Harga Bitcoin (BTC)
    Model menggunakan algoritma **LSTM (Keras)**. Memprediksi pergerakan harga 7 jam ke depan berdasarkan rentetan data harga 24 jam terakhir.
    👉 Buka menu **P2 App** di sidebar.
    """
)
