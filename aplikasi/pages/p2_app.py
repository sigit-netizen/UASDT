import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import MetaTrader5 as mt5
from datetime import datetime, timezone

# ==========================================
# 1. PERSIAPAN APLIKASI & MODEL AI
# ==========================================
st.title("P2: Forecasting Harga Bitcoin (BTC)")
st.write("Model LSTM memprediksi harga BTC untuk **7 jam ke depan** berdasarkan data **24 jam terakhir**.")

# Memuat model dan scaler (disimpan dalam cache agar tidak diload berulang kali)
@st.cache_resource
def load_models():
    try:
        from tensorflow.keras.models import load_model
        base_path = os.path.join(os.path.dirname(__file__), '../../P2/model')
        lstm_model = load_model(os.path.join(base_path, 'lstm_forecasting.keras'))
        with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        return lstm_model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

lstm_model, scaler = load_models()


# ==========================================
# 2. FUNGSI KHUSUS UNTUK MEMBANTU PROSES
# ==========================================

def get_mt5_data(symbol, target_date, target_time):
    """Menarik 24 data harga penutupan (Close) H1 terakhir dari MT5"""
    if not mt5.initialize():
        st.error(f"Gagal konek ke MT5. Error: {mt5.last_error()}")
        return None
    
    # Gabungkan tanggal & waktu, lalu amankan ke zona waktu UTC+0 (Waktu Server MT5)
    dt_target = datetime.combine(target_date, target_time).replace(tzinfo=timezone.utc)
    
    # Tarik 24 data mundur dari waktu target
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, dt_target, 24)
    mt5.shutdown()
    
    if rates is None or len(rates) < 24:
        return None
        
    # Kembalikan daftar angka harga saja (dibulatkan 2 desimal)
    return [round(r['close'], 2) for r in rates]


def run_lstm_prediction(prices, model, scaler):
    """Menebak harga 7 jam ke depan menggunakan AI LSTM"""
    # 1. Sesuaikan skala angka (Preprocessing) agar ramah untuk AI
    input_array = np.array(prices).reshape(-1, 1)
    scaled_input = scaler.transform(input_array)
    
    # 2. Ubah ke bentuk matriks 3 Dimensi untuk LSTM: (batch=1, time_steps=24, features=1)
    current_batch = scaled_input.reshape(1, 24, 1) 
    
    forecast_list = []
    
    # 3. Lakukan perulangan tebakan untuk 7 jam ke depan
    for _ in range(7):
        # AI Menebak 1 angka ke depan
        pred = model.predict(current_batch, verbose=0)
        pred_scalar = float(pred.flatten()[0])
        forecast_list.append(pred_scalar)
        
        # Masukkan hasil tebakan baru ini ke ujung gerbong antrean, 
        # sekaligus membuang 1 data terlama di awal antrean (Teknik Sliding Window)
        new_step = np.array([[[pred_scalar]]])
        current_batch = np.append(current_batch[:, 1:, :], new_step, axis=1)
        
    # 4. Kembalikan angka tebakan ke skala Dolar aslinya
    return scaler.inverse_transform(np.array(forecast_list).reshape(-1, 1)).flatten()


# ==========================================
# 3. TAMPILAN ANTARMUKA (SUMBER DATA)
# ==========================================
st.markdown("---")
st.subheader("1. Sumber Data (24 Jam Terakhir)")

col1, col2, col3 = st.columns(3)
symbol = col1.text_input("Simbol MT5", value="BTCUSDc")
target_date = col2.date_input("Tanggal", value=datetime.today())

# Waktu otomatis digenapkan ke jam bulat terdekat (contoh 11:05 -> 11:00)
rounded_time = datetime.now().time().replace(minute=0, second=0, microsecond=0)
target_time = col3.time_input("Waktu MT5 (UTC+0)", value=rounded_time, step=3600)

col_btn1, col_btn2 = st.columns(2)

# ---- Tombol Tarik MT5 ----
if col_btn1.button("🔄 Tarik Data MT5 Otomatis", use_container_width=True):
    with st.spinner("Menyedot data dari MT5..."):
        prices = get_mt5_data(symbol, target_date, target_time)
        if prices:
            # Jika berhasil, simpan deretan angka ke penyimpanan layar (session_state)
            st.session_state['input_prices'] = ", ".join(map(str, prices))
            st.success("Sukses ditarik!")
        else:
            st.error("Gagal menarik data. Pastikan simbol benar.")

# ---- Tombol Generate Dummy ----
if col_btn2.button("🎲 Buat Data Acak (Dummy)", use_container_width=True):
    dummy_prices = [60000.0]
    for _ in range(23): # Buat sisa 23 angka acak layaknya pergerakan saham
        dummy_prices.append(round(dummy_prices[-1] + np.random.normal(0, 200), 2))
    st.session_state['input_prices'] = ", ".join(map(str, dummy_prices))


# ==========================================
# 4. FORM INPUT & TOMBOL PREDIKSI
# ==========================================
st.markdown("---")
st.subheader("2. Hasil Input & Prediksi")

# Tampilkan isi text box berdasarkan tarikan MT5/Dummy (jika kosong, tampilkan string kosong)
user_input = st.text_area(
    "Deretan 24 Angka Harga (Pisahkan dengan koma):", 
    value=st.session_state.get('input_prices', ''), 
    height=100
)

if st.button("🚀 Jalankan Prediksi", type="primary"):
    if not lstm_model:
        st.error("Model AI belum termuat.")
    elif not user_input:
        st.warning("Kotak input masih kosong.")
    else:
        try:
            # Mengubah teks panjang menjadi daftar angka (memisahkan koma & membuang spasi kosong)
            prices = [float(x.strip()) for x in user_input.split(',') if x.strip() != '']
            
            if len(prices) != 24:
                st.warning(f"Jumlah angka harus tepat 24! Saat ini ada {len(prices)}.")
            else:
                with st.spinner("AI sedang menghitung prediksi..."):
                    # Panggil fungsi tebakan (LSTM)
                    forecast_prices = run_lstm_prediction(prices, lstm_model, scaler)
                
                st.success("Hitungan Selesai!")
                
                # ==========================================
                # 5. MENGGAMBAR GRAFIK & TABEL
                # ==========================================
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Menggambar Garis Biru (24 Jam Masa Lalu)
                ax.plot(range(1, 25), prices, label='Aktual (24 Jam)', marker='o', color='blue')
                
                # Menggambar Garis Oranye (7 Jam Masa Depan) 
                # Angka masa lalu terakhir disambung agar garis grafiknya menyatu
                plot_times = list(range(24, 32))
                plot_prices = [prices[-1]] + list(forecast_prices)
                ax.plot(plot_times, plot_prices, label='Prediksi (7 Jam)', marker='x', color='orange', linestyle='--')
                
                ax.set_title(f"Forecasting Harga {symbol}")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig) # Tampilkan di layar
                
                # Menampilkan Tabel Prediksi yang rapi
                st.dataframe(pd.DataFrame({
                    'Jam Ke Depan': [f"+{i+1} jam" for i in range(7)],
                    'Harga (USD)': [f"${p:,.2f}" for p in forecast_prices]
                }))
                
        except ValueError as ve:
            st.error(f"Terdapat huruf atau titik dua di input yang tidak bisa dibaca komputer. Error detail: {ve}")
