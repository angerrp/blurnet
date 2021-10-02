import csv
import os
import random

dirs = {0: ["/home/paul/Downloads/blur-dataset/sharp"],
        1: ["/home/paul/Downloads/blur-dataset/motion_blurred", "/home/paul/Downloads/blur-dataset/defocused_blurred"]}
img_dir = "/home/paul/Downloads/blur-dataset/sharp"

header = []
data = []
test_data = []
for key, value in dirs.items():
    for val in value:
        files = os.listdir(val)
        for file in os.listdir(val):
            data.append([file, key])
random.shuffle(data)
for i in range(round(int(len(data) / 5))):
    test_data.append(data[i])

with open('/home/paul/code/blur_detection/data_binary_classification.csv', 'w') as f:
    writer = csv.writer(f)
    # writer.writerow(header)
    writer.writerows(data)
