import numpy as np
from sklearn.metrics import davies_bouldin_score
import os

input_dir = r"E:\Date\Teste\7_MDPI\Data\Full"
label_dir = r'E:\Date\Teste\7_MDPI\Labels\DWT\500_Resized'
filename_list = []
davies_bouldin_total = 0
count = 0
for filename in os.listdir(input_dir):
    input_data = np.load(os.path.join(input_dir, filename))
    input_data = input_data.reshape(-1, 4)
    label_data = np.load(os.path.join(label_dir, filename))
    label_data = label_data.reshape(-1, 1)
    label_data = label_data.ravel()

    davies_bouldin = davies_bouldin_score(input_data, label_data)
    davies_bouldin_total = davies_bouldin_total + davies_bouldin

    count += 1
    print(count, davies_bouldin)

davies_bouldin_avg = davies_bouldin_total / count
print("DBI Average = ", davies_bouldin_avg)
