# UAS Data Mining


## 📂 Lokasi Dataset Utama & Hasil Split

### Proyek 1 (P1): US Accidents March 2023(https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
* **Dataset Asli (Raw Data)**
  Pastikan Anda meletakkan file `US_Accidents_March23.csv` (berukuran ~3GB) di dalam lokasi berikut:
  `P2/dataset/US_Accidents_March23.csv`
* **Data Hasil Split (Train/Test)**
  Folder penyimpanannya ada di: `P2/dataset/data/`.
  File hasil split (`train.csv` dan `test.csv`) akan otomatis muncul di folder ini setelah Anda menjalankan script `split.py`.

---

### Proyek 2 (P2): Data Historis Bitcoin(https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
* **Dataset Asli (Raw Data)**
  Pastikan Anda meletakkan file `Data Historis Bitcoin` di dalam lokasi berikut:
  `P1/dataset/Data Historis Bitcoin`
* **Data Hasil Split (Train/Test)**
  Folder penyimpanannya ada di: `P1/dataset/data/`.
  File hasil split (`train.csv` dan `test.csv`) akan otomatis muncul di folder ini setelah Anda menjalankan script `split.py`.



## ⚙️ Cara Menyiapkan Data (Proses Pemotongan / Split)

Jika dataset asli sudah diletakkan di dalam folder `dataset` masing-masing, ikuti langkah berikut untuk membagi data menjadi 80% Training dan 20% Testing:

1. Buka Terminal/Command Prompt di komputer lokal.
2. Pindah ke direktori training:
   ```bash
   cd P1/training
   # atau
   cd P2/training
   ```
3. Jalankan script python untuk memotong data:
   ```bash
   python split.py
   ```
4. Setelah script selesai, silakan cek folder `dataset/data/` pada masing-masing proyek untuk memastikan file `train.csv` dan `test.csv` sudah terbentuk.