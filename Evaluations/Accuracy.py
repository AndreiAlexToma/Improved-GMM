import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from memory_profiler import profile
import time
import os

# input_data = np.load(r"E:\Date\Teste\7_MDPI\Training image\GMM_5\Data.npy")
# input_cut = input_data[750:, 750:, 1:4]
# cv2.imwrite(r"E:\Date\Teste\7_MDPI\Training image\Manual_labeling\Ground_truth_250.TIF", input_cut)
#
# output_data = cv2.imread(r"E:\Date\Teste\7_MDPI\Training image\GMM_5\GMM_5.tif")
# output_cut = output_data[750:, 750:, :]
# cv2.imwrite(r"E:\Date\Teste\7_MDPI\Training image\Manual_labeling\Output_250.TIF", output_cut)

label_data = np.load(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DFT\Labels_5_DFT_250_resized.npy")
pred_labels = label_data[750:, 750:]
np.save(r"E:\Date\Teste\7_MDPI\Training image\Manual_labeling\Labels_5_DFT_250_CUT.npy", pred_labels)

mask = (pred_labels == 0) | (pred_labels == 4)
pred_labels[mask] = np.where(pred_labels[mask] == 0, 4, 0)
mask = (pred_labels == 2) | (pred_labels == 3)
pred_labels[mask] = np.where(pred_labels[mask] == 2, 3, 2)
mask = (pred_labels == 2) | (pred_labels == 4)
pred_labels[mask] = np.where(pred_labels[mask] == 2, 4, 2)
# mask = (pred_labels == 2) | (pred_labels == 3)
# pred_labels[mask] = np.where(pred_labels[mask] == 2, 3, 2)



count_0 = np.count_nonzero(pred_labels == 0)
count_1 = np.count_nonzero(pred_labels == 1)
count_2 = np.count_nonzero(pred_labels == 2)
count_3 = np.count_nonzero(pred_labels == 3)
count_4 = np.count_nonzero(pred_labels == 4)
print(count_0)
print(count_1)
print(count_2)
print(count_3)
print(count_4)
acc_0 = count_0 / 15730
acc_1 = 12475 / count_1
acc_2 = count_2 / 14320
acc_3 = 6942 / count_3
acc_4 = count_4 / 13032

sum = (acc_4 + acc_3 + acc_2 + acc_1 + acc_0)/5

print(acc_0)
print(acc_1)
print(acc_2)
print(acc_3)
print(acc_4)
print("mean=", sum)

# true_labels = np.load(r"E:\Date\Teste\7_MDPI\Training image\GMM_5\Labels_5.npy")
# pred_labels = np.load(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DCT\Labels_5_DCT_200_resized.npy")
#
# mask = (pred_labels == 0) | (pred_labels == 1)
# pred_labels[mask] = np.where(pred_labels[mask] == 0, 1, 0)
# mask = (pred_labels == 1) | (pred_labels == 3)
# pred_labels[mask] = np.where(pred_labels[mask] == 1, 3, 1)
# mask = (pred_labels == 1) | (pred_labels == 4)
# pred_labels[mask] = np.where(pred_labels[mask] == 1, 4, 1)
# # mask = (pred_labels == 1) | (pred_labels == 4)
# # pred_labels[mask] = np.where(pred_labels[mask] == 1, 4, 1)
#
#
# num_classes = 5
# # Initialize an array to store the count of correct predictions for each class
# correct_per_class = np.zeros(num_classes)
#
# # Iterate through each class
# for i in range(num_classes):
#     # Create a mask for samples where the true label is equal to the current class
#     class_mask = (true_labels == i)
#     # Count the number of correct predictions for the current class
#     correct_predictions = np.sum(pred_labels[class_mask] == true_labels[class_mask])
#
#     # Calculate the accuracy for the current class
#     total_samples = np.sum(class_mask)
#     if total_samples > 0:
#         accuracy = correct_predictions / total_samples
#         correct_per_class[i] = accuracy
#
#     print(i, correct_per_class[i])