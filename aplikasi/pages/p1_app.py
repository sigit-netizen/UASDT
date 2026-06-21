import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

st.title("P1: Prediksi Tingkat Keparahan Kecelakaan")

base_path = os.path.join(os.path.dirname(__file__), '../../P1/model')

@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load(os.path.join(base_path, 'random_forest.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        prep = joblib.load(os.path.join(base_path, 'preprocessing.pkl'))
        return rf_model, scaler, prep
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

rf_model, scaler, prep = load_models()

input_mode = st.radio("Pilih Metode Input Data:", ["Otomatis (Generate Random)", "Manual Input"], horizontal=True)

if input_mode == "Otomatis (Generate Random)":
    st.write("Karena dataset asli sangat besar, Anda dapat men-generate data sampel (dummy) secara acak.")
    if st.button("Generate Random Data & Prediksi"):
        if rf_model is not None and prep is not None:
            try:
                dummy_data = {}
                numeric_cols = list(prep['median_impute'].keys())
                categorical_cols = list(prep['mode_impute'].keys())
                
                for col in numeric_cols:
                    if col != 'Severity':
                        dummy_data[col] = prep['median_impute'][col] + np.random.normal(0, 1)
                    else:
                        dummy_data[col] = 0
                        
                for col in categorical_cols:
                    dummy_data[col] = 0
                
                if hasattr(scaler, 'feature_names_in_'):
                    expected_cols = scaler.feature_names_in_
                else:
                    expected_cols = numeric_cols + categorical_cols
                
                df_dummy = pd.DataFrame([dummy_data], columns=expected_cols).fillna(0)
                X_scaled = scaler.transform(df_dummy)
                
                if hasattr(rf_model, 'n_features_in_') and X_scaled.shape[1] < rf_model.n_features_in_:
                    diff = rf_model.n_features_in_ - X_scaled.shape[1]
                    X_scaled = np.append(X_scaled, np.zeros((X_scaled.shape[0], diff)), axis=1)
                
                prediction = rf_model.predict(X_scaled)
                st.success(f"**Prediksi Tingkat Keparahan (Severity): {prediction[0]}**")
                
                with st.expander("Lihat detail input data dummy"):
                    st.dataframe(df_dummy)
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memprediksi: {e}")
                
else: # Manual Input
    st.write("Silakan isi data kecelakaan sesuai informasi yang Anda ketahui. Nilai default sudah diisikan untuk parameter yang tidak Anda ketahui.")
    
    with st.form("manual_form"):
        numeric_cols = list(prep['median_impute'].keys())
        categorical_cols = list(prep['mode_impute'].keys())
        
        if hasattr(scaler, 'feature_names_in_'):
            expected_cols = scaler.feature_names_in_
        else:
            expected_cols = numeric_cols + categorical_cols
            
        manual_data = {}
        
        st.write("### Form Input (44 Parameter)")
        cols = st.columns(4) # Bikin 4 kolom biar rapi
        
        for i, col in enumerate(expected_cols):
            with cols[i % 4]:
                if col in numeric_cols:
                    default_val = float(prep['median_impute'][col])
                    manual_data[col] = st.number_input(col, value=default_val, key=col)
                elif col in categorical_cols:
                    if col in prep['label_encoders']:
                        classes = prep['label_encoders'][col]['classes']
                        if len(classes) <= 100:
                            # Jika class sedikit, pakai selectbox
                            selected = st.selectbox(col, options=classes, key=col)
                            manual_data[col] = prep['label_encoders'][col]['mapping'].get(selected, 0)
                        else:
                            # Jika class ribuan (seperti Start_Time), pakai text input untuk mencegah browser hang
                            val = st.text_input(col, value=str(classes[0]), help="Input teks biasa", key=col)
                            manual_data[col] = prep['label_encoders'][col]['mapping'].get(val, 0)
                    else:
                        manual_data[col] = 0
                else:
                    # Sisa kolom yang tidak ada di prep (misal fitur Boolean seperti Bump, Crossing, dll)
                    val = st.selectbox(col, options=[False, True], key=col)
                    manual_data[col] = 1 if val else 0
                    
        submitted = st.form_submit_button("Prediksi Manual")
        
        if submitted:
            if rf_model is not None and prep is not None:
                try:
                    df_manual = pd.DataFrame([manual_data], columns=expected_cols).fillna(0)
                    X_scaled = scaler.transform(df_manual)
                    
                    if hasattr(rf_model, 'n_features_in_') and X_scaled.shape[1] < rf_model.n_features_in_:
                        diff = rf_model.n_features_in_ - X_scaled.shape[1]
                        X_scaled = np.append(X_scaled, np.zeros((X_scaled.shape[0], diff)), axis=1)
                    
                    prediction = rf_model.predict(X_scaled)
                    st.success(f"**Prediksi Tingkat Keparahan (Severity): {prediction[0]}**")
                    
                    with st.expander("Lihat detail 44 parameter input"):
                        st.dataframe(df_manual)
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memprediksi: {e}")
                    st.write("Pastikan environment Python Anda menggunakan versi numpy yang kompatibel (numpy < 2.0).")
            else:
                st.error("Model belum dimuat. Periksa ketersediaan file pkl.")
