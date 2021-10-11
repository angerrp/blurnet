import csv
import os
import random
import shutil

blur_dataset = "/home/paul/Downloads/blur-dataset"
merged_dir = "tmp"
test_dir = "test"
dirs = {0: [os.path.join(blur_dataset, "sharp")],
        1: [os.path.join(blur_dataset, "motion_blurred")]}

header = []
data = []
test_data = []
overall_files = []
for key, value in dirs.items():
    for val in value:
        files = os.listdir(val)
        for file in files:
            data.append([file, key])
random.shuffle(data)
train_split = int(len(data) * 0.8)
train_data = []
train_labels = []
for i in range(train_split):
    train_data.append(data[i])

for i in range(train_split, len(data)):
    test_data.append(data[i])
  #  shutil.move(os.path.join(blur_dataset, merged_dir, data[i][0]), os.path.join(blur_dataset, test_dir))

with open('motion_test_set.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(test_data)

with open('motion_train_set.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train_data)
