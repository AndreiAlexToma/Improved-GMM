import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import time
import os

count_images = 0
total_time = 0
total_processing_time = 0
total_training_time = 0
total_prediction_time = 0
dirname = r'E:\Date\Teste\3_ISMSIT Turcia 2022\final_results\Input\test\4000\RGBN'
for fname in os.listdir(dirname):

    t0 = time.time()

    image = np.load(os.path.join(dirname, fname))
    pixel_values = image.reshape((-1, 4))
    pixel_values = np.float32(pixel_values)

    t1 = time.time()
    processing_time_seconds = t1 - t0

    t2 = time.time()

    clustering = GaussianMixture(n_components=2, random_state=42)
    clustering.fit(pixel_values)

    t3 = time.time()
    training_time_seconds = t3 - t2

    t4 = time.time()

    labels = clustering.predict(pixel_values)
    centers = clustering.means_
    centers = np.uint8(centers)
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(image.shape)
    segmented_image = segmented_image[:, :, 1:4]
    # cv2.imwrite(r"E:\Date\Teste\7_MDPI\Output\Full/" + os.path.splitext(fname)[0] + ".TIF", segmented_image)

    t5 = time.time()
    prediction_time_seconds = t5 - t4

    labels = labels.reshape((image.shape[0], image.shape[1]))
    # np.save(r"E:\Date\Teste\7_MDPI\Labels\Full/" + os.path.splitext(fname)[0] + ".npy", labels)

    count_images += 1
    total_processing_time = total_processing_time + processing_time_seconds
    total_training_time = total_training_time + training_time_seconds
    total_prediction_time = total_prediction_time + prediction_time_seconds
    print(count_images)

mean_processing_time = total_processing_time / count_images
mean_training_time = total_training_time / count_images
mean_prediction_time = total_prediction_time / count_images

print("Mean processing time = ", mean_processing_time)
print("Mean training time = ", mean_training_time)
print("Mean prediction time = ", mean_prediction_time)
