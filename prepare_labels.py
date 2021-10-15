import argparse
import csv
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to dataset.")
    args = parser.parse_args()
    blur_dataset = args.path
    dirs = {
        0: [os.path.join(blur_dataset, "sharp")],
        1: [
            os.path.join(blur_dataset, "motion_blurred"),
            os.path.join(blur_dataset, "defocused_blurred"),
        ],
    }

    data = []
    test_data = []
    for key, value in dirs.items():
        for val in value:
            files = os.listdir(val)
            for file in files:
                data.append([os.path.join(val, file), key])

    random.shuffle(data)
    train_split = int(len(data) * 0.8)
    train_data = []
    for i in range(train_split):
        train_data.append(data[i])

    for i in range(train_split, len(data)):
        test_data.append(data[i])

    with open("test_set.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(test_data)

    with open("train_set.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(train_data)


if __name__ == "__main__":
    main()
