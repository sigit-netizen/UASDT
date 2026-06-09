import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

def main():
    BASE_DIR = Path(__file__).resolve().parent

    csv_path = BASE_DIR.parent / 'dataset' / 'train.csv'
    output_dir = BASE_DIR.parent / 'dataset' / 'data'
    
    print("Membaca dataset train.csv...")
    df = pd.read_csv(csv_path)
    
    print("Membagi dataset menjadi 80% training dan 20% testing...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Menyimpan data ke dalam folder dataset/data...")
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print("Selesai!")

if __name__ == '__main__':
    main()
