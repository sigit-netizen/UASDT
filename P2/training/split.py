import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    csv_path = '../dataset/air_pollution_china.csv'
    output_dir = '../dataset/data'
    
    print("Membaca dataset air_pollution_china.csv...")
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
