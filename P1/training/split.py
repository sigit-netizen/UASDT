import csv
import random
import os
import sys

def main():
    csv_path = '../dataset/US_Accidents_March23.csv'
    output_dir = '../dataset/data'
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    print("Membaca dan membagi dataset US_Accidents_March23 secara efisien (baris per baris)...")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f_in, \
             open(train_path, 'w', encoding='utf-8', newline='') as f_train, \
             open(test_path, 'w', encoding='utf-8', newline='') as f_test:
            
            reader = csv.reader(f_in)
            writer_train = csv.writer(f_train)
            writer_test = csv.writer(f_test)
            
            # Tulis header
            headers = next(reader)
            writer_train.writerow(headers)
            writer_test.writerow(headers)
            
            # Set seed agar hasil pembagian selalu sama
            random.seed(42)
            
            count = 0
            for row in reader:
                # 80% probabilitas masuk train, 20% masuk test
                if random.random() < 0.8:
                    writer_train.writerow(row)
                else:
                    writer_test.writerow(row)
                
                count += 1
                if count % 500000 == 0:
                    print(f"Sudah memproses {count} baris...")
                    
        print("Selesai! Data berhasil dibagi dan disimpan ke dataset/data.")
        
    except Exception as e:
        print(f"Terjadi error: {e}")

if __name__ == '__main__':
    main()
