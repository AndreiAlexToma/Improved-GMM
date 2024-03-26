import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from memory_profiler import profile
import time
import pywt


@profile()
def main():

    t0 = time.time()

    image = np.load(r"E:\Date\Teste\7_MDPI\Training image\Data.npy")
    wavelet = 'haar'
    nir_coeffs, _ = pywt.wavedec2(image[:, :, 0], wavelet, level=1)
    red_coeffs, _ = pywt.wavedec2(image[:, :, 1], wavelet, level=1)
    gre_coeffs, _ = pywt.wavedec2(image[:, :, 2], wavelet, level=1)
    blu_coeffs, _ = pywt.wavedec2(image[:, :, 3], wavelet, level=1)
    nir_coeffs_2, _ = pywt.wavedec2(nir_coeffs, wavelet, level=1)
    red_coeffs_2, _ = pywt.wavedec2(red_coeffs, wavelet, level=1)
    gre_coeffs_2, _ = pywt.wavedec2(gre_coeffs, wavelet, level=1)
    blu_coeffs_2, _ = pywt.wavedec2(blu_coeffs, wavelet, level=1)
    red_coeffs_2_aprox = np.expand_dims(red_coeffs_2 / red_coeffs_2.max() * 255, axis=-1)
    gre_coeffs_2_aprox = np.expand_dims(gre_coeffs_2 / gre_coeffs_2.max() * 255, axis=-1)
    blu_coeffs_2_aprox = np.expand_dims(blu_coeffs_2 / blu_coeffs_2.max() * 255, axis=-1)
    nir_coeffs_2_aprox = np.expand_dims(nir_coeffs_2 / nir_coeffs_2.max() * 255, axis=-1)
    inverse = np.concatenate((nir_coeffs_2_aprox, red_coeffs_2_aprox, gre_coeffs_2_aprox, blu_coeffs_2_aprox), axis=-1)
    inverse[:, :, 0] = inverse[:, :, 0] + np.mean(image[:, :, 0]) - np.mean(inverse[:, :, 0])
    inverse[:, :, 1] = inverse[:, :, 1] + np.mean(image[:, :, 1]) - np.mean(inverse[:, :, 1])
    inverse[:, :, 2] = inverse[:, :, 2] + np.mean(image[:, :, 2]) - np.mean(inverse[:, :, 2])
    inverse[:, :, 3] = inverse[:, :, 3] + np.mean(image[:, :, 3]) - np.mean(inverse[:, :, 3])
    inverse = np.uint8(inverse)
    np.save(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DWT\Data_DWT_250.npy", inverse)
    pixel_values = inverse.reshape(-1, 4)
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
    cv2.imwrite(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DWT\GMM_DWT_250.TIF", segmented_image)

    t5 = time.time()
    prediction_time_seconds = t5 - t4

    print("Processing time = ", processing_time_seconds)
    print("Training time = ", training_time_seconds)
    print("Prediction time = ", prediction_time_seconds)

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    labels = labels.reshape((inverse.shape[0], inverse.shape[1]))
    np.save(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DWT\Labels_5_DWT_250.npy", labels)
    labels = cv2.resize(labels, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    np.save(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DWT\Labels_5_DWT_250_resized.npy", labels)

    image = image[:, :, 1:4]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(image)
    # axarr[1].imshow(segmented_image)
    # plt.show()


if __name__ == "__main__":
    main()


