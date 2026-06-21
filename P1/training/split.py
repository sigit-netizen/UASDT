import csv
import random
import os

def binarize_severity(severity):
    severity = int(severity)
    return 0 if severity in (1, 2) else 1

def main():
    csv_path = '../dataset/US_Accidents_March23.csv'
    output_dir = '../dataset/data'

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    random.seed(42)

    # --- PASS 1: Hitung distribusi kelas di train set ---
    print("Pass 1: Menghitung distribusi kelas...")
    train_counts = {0: 0, 1: 0}
    total_rows = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        sev_idx = headers.index('Severity')

        for row in reader:
            total_rows += 1
            if random.random() < 0.8:
                cls = binarize_severity(row[sev_idx])
                train_counts[cls] += 1

            if total_rows % 1000000 == 0:
                print(f"  Pass 1: {total_rows} baris diproses...")

    minority = 0 if train_counts[0] < train_counts[1] else 1
    majority = 1 - minority
    majority_sample_rate = train_counts[minority] / train_counts[majority]

    print(f"  Train class 0 (minor 1-2): {train_counts[0]}")
    print(f"  Train class 1 (severe 3-4): {train_counts[1]}")
    print(f"  Minority class: {minority}, Majority: {majority}")
    print(f"  Sampling rate for majority: {majority_sample_rate:.4f}")

    # --- PASS 2: Tulis train (balanced) dan test ---
    print("\nPass 2: Menulis dataset (train balanced)...")
    random.seed(42)

    with open(csv_path, 'r', encoding='utf-8') as f_in, \
         open(train_path, 'w', encoding='utf-8', newline='') as f_train, \
         open(test_path, 'w', encoding='utf-8', newline='') as f_test:

        reader = csv.reader(f_in)
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)

        headers = next(reader)
        sev_idx = headers.index('Severity')
        writer_train.writerow(headers)
        writer_test.writerow(headers)

        written_train = {0: 0, 1: 0}
        count = 0

        for row in reader:
            cls = binarize_severity(row[sev_idx])
            row[sev_idx] = str(cls)

            is_train = random.random() < 0.8

            if is_train:
                if cls == majority and random.random() > majority_sample_rate:
                    pass
                else:
                    writer_train.writerow(row)
                    written_train[cls] += 1
            else:
                writer_test.writerow(row)

            count += 1
            if count % 500000 == 0:
                print(f"  Pass 2: {count} baris diproses...")

    print(f"\nSelesai!")
    print(f"  Train: class 0 = {written_train[0]}, class 1 = {written_train[1]}")
    print(f"  Test: {count - sum(written_train.values())} rows (all classes)")

if __name__ == '__main__':
    main()
