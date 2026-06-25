import csv
import random
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, 'Dataset', 'US_Accidents_March23.csv')
    output_dir = os.path.join(base_dir, 'Dataset', 'data')

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    random.seed(42)

    print("Membagi dataset menjadi 80% training dan 20% testing...")

    with open(csv_path, 'r', encoding='utf-8') as f_in, \
         open(train_path, 'w', encoding='utf-8', newline='') as f_train, \
         open(test_path, 'w', encoding='utf-8', newline='') as f_test:

        reader = csv.reader(f_in)
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)

        headers = next(reader)
        writer_train.writerow(headers)
        writer_test.writerow(headers)

        count = 0
        for row in reader:
            if random.random() < 0.8:
                writer_train.writerow(row)
            else:
                writer_test.writerow(row)

            count += 1
            if count % 500000 == 0:
                print(f"  {count} baris diproses...")

    print(f"\nSelesai! Train = ~{count * 0.8:.0f}, Test = ~{count * 0.2:.0f} rows")

if __name__ == '__main__':
    main()
