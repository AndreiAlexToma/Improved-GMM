import numpy as np
from sklearn.metrics import davies_bouldin_score

input_data = np.load(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DFT\Data_DFT_125.npy")
label_data = np.load(r'E:\Date\Teste\7_MDPI\Training image\GMM_5_DFT\Labels_5_DFT_125.npy')

input_data = input_data.reshape(-1, 4)
label_data = label_data.reshape(-1, 1)
label_data = label_data.ravel()

davies_bouldin = davies_bouldin_score(input_data, label_data)

print(davies_bouldin)
