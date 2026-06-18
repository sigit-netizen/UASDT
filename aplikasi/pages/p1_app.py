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

st.write("Karena dataset asli sangat besar (3GB) dan form ini membutuhkan 45 parameter, Anda dapat men-generate data sampel (dummy) untuk melihat bagaimana model memprediksi severity.")

if st.button("Generate Random Data & Prediksi"):
    if rf_model is not None and prep is not None:
        try:
            # Menggunakan feature names dari prep dictionary
            feature_names = prep.get('target_column', None) # target column is saved, features might be in a list
            
            # Create a dummy row using median and mode from prep
            dummy_data = {}
            numeric_cols = list(prep['median_impute'].keys())
            categorical_cols = list(prep['mode_impute'].keys())
            
            # Fill numerics with medians + some random noise
            for col in numeric_cols:
                median_val = prep['median_impute'][col]
                dummy_data[col] = median_val + np.random.normal(0, 1)
                
            # Fill categoricals with first available encoded class
            for col in categorical_cols:
                if col in prep['label_encoders']:
                    # Get the encoded integer for the most common class or 0
                    dummy_data[col] = 0
                else:
                    dummy_data[col] = 0
            
            # It expects the columns in the exact order as training.
            # Usually we can get the expected features from scaler.feature_names_in_ if using sklearn >= 1.0
            if hasattr(scaler, 'feature_names_in_'):
                expected_cols = scaler.feature_names_in_
            else:
                expected_cols = numeric_cols + categorical_cols # fallback

            # Construct DataFrame with exactly the expected columns
            df_dummy = pd.DataFrame([dummy_data], columns=expected_cols).fillna(0)
            
            # Scale the data
            X_scaled = scaler.transform(df_dummy)
            
            # Predict
            prediction = rf_model.predict(X_scaled)
            
            st.success(f"**Prediksi Tingkat Keparahan (Severity): {prediction[0]}**")
            
            with st.expander("Lihat detail input data dummy"):
                st.dataframe(df_dummy)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}")
            st.write("Pastikan environment Python Anda menggunakan versi numpy yang kompatibel (numpy < 2.0).")
    else:
        st.error("Model belum dimuat. Periksa ketersediaan file pkl.")
