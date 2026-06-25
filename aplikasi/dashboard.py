import streamlit as st

st.set_page_config(
    page_title="Prediksi & Forecasting Dashboard",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>Dashboard Cerdas Prediksi Risiko Kecelakaan dan Forecasting Aset Digital</h1>", unsafe_allow_html=True)
st.divider()

st.markdown("<p style='text-align: center; font-size: 18px; margin-bottom: 30px;'>Aplikasi ini dibangun untuk menggabungkan dua fungsionalitas:</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.info("**Proyek 1**")
    st.markdown("### Klasifikasi Tingkat Keparahan Kecelakaan (US Accidents)")
    st.markdown("Model menggunakan algoritma **Random Forest**. Memprediksi tingkat keparahan berdasarkan fitur-fitur kondisi lingkungan, lokasi, dll.")
    st.markdown("**Buka menu *prediksi_kecelakakan* di sidebar.**")

with col2:
    st.success("**Proyek 2**")
    st.markdown("### Forecasting Harga Bitcoin (BTC)")
    st.markdown("Model menggunakan algoritma **LSTM (Keras)**. Memprediksi pergerakan harga 7 jam ke depan berdasarkan rentetan data harga 24 jam terakhir.")
    st.markdown("**Buka menu *forecasting_btC* di sidebar.**")

st.divider()

# Tempatkan kembali identitas yang terhapus agar laporan aman
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small><i>Aplikasi ini adalah hasil karya orisinil dan dibuat secara unik untuk memenuhi syarat UAS mata kuliah Data Mining.</i></small><br>
    <small><b>Identitas:</b></small>
    <br><small>Faizin Hilal: 2313020024</small><br><small>Mohammad Rizky Andi Putra: 2313020229</small><br><small>Sigit: 2313020027</small><br><small>
    </div>
    """,
    unsafe_allow_html=True
)
