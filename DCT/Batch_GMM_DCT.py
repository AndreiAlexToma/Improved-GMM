import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from memory_profiler import profile
import time
from scipy.fftpack import dctn, idctn
import os

count_images = 0
total_processing_time = 0
total_training_time = 0
total_prediction_time = 0
dirname = r'E:\Date\Teste\7_MDPI\Data\Full'
for fname in os.listdir(dirname):

    t0 = time.time()

    image = np.load(os.path.join(dirname, fname))
    k = 125
    image_dct_nir = idctn(dctn(image[:, :, 0], norm="ortho")[:k, :k], norm="ortho")
    image_dct_red = idctn(dctn(image[:, :, 1], norm="ortho")[:k, :k], norm="ortho")
    image_dct_gre = idctn(dctn(image[:, :, 2], norm="ortho")[:k, :k], norm="ortho")
    image_dct_blu = idctn(dctn(image[:, :, 3], norm="ortho")[:k, :k], norm="ortho")
    normalized_nir = np.expand_dims(image_dct_nir / image_dct_nir.max() * 255, axis=-1)
    normalized_red = np.expand_dims(image_dct_red / image_dct_red.max() * 255, axis=-1)
    normalized_gre = np.expand_dims(image_dct_gre / image_dct_gre.max() * 255, axis=-1)
    normalized_blu = np.expand_dims(image_dct_blu / image_dct_blu.max() * 255, axis=-1)
    inverse = np.concatenate((normalized_nir, normalized_red, normalized_gre, normalized_blu), axis=-1)
    inverse[:, :, 0] = inverse[:, :, 0] + np.mean(image[:, :, 0]) - np.mean(inverse[:, :, 0])
    inverse[:, :, 1] = inverse[:, :, 1] + np.mean(image[:, :, 1]) - np.mean(inverse[:, :, 1])
    inverse[:, :, 2] = inverse[:, :, 2] + np.mean(image[:, :, 2]) - np.mean(inverse[:, :, 2])
    inverse[:, :, 3] = inverse[:, :, 3] + np.mean(image[:, :, 3]) - np.mean(inverse[:, :, 3])
    inverse = np.uint8(inverse)

    np.save(r"E:\Date\Teste\7_MDPI\Data\DCT\125/" + os.path.splitext(fname)[0] + ".npy", inverse)
    pixel_values = inverse.reshape((-1, 4))
    pixel_values = np.float32(pixel_values)

    t1 = time.time()
    processing_time_seconds = t1 - t0

    t2 = time.time()

    clustering = GaussianMixture(n_components=5, random_state=42)
    clustering.fit(pixel_values)

    t3 = time.time()
    training_time_seconds = t3 - t2

    t4 = time.time()

    labels = clustering.predict(pixel_values)
    centers = clustering.means_
    centers = np.uint8(centers)
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(inverse.shape)
    segmented_image = cv2.resize(segmented_image, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    segmented_image = segmented_image[:, :, 1:4]
    cv2.imwrite(r"E:\Date\Teste\7_MDPI\Output\DCT\125_Resized/" + os.path.splitext(fname)[0] + ".TIF", segmented_image)

    t5 = time.time()
    prediction_time_seconds = t5 - t4

    labels = labels.reshape((inverse.shape[0], inverse.shape[1]))
    np.save(r"E:\Date\Teste\7_MDPI\Labels\DCT\125/" + os.path.splitext(fname)[0] + ".npy", labels)
    labels = cv2.resize(labels, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    np.save(r"E:\Date\Teste\7_MDPI\Labels\DCT\125_Resized/" + os.path.splitext(fname)[0] + ".npy", labels)

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
