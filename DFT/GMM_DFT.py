import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from memory_profiler import profile
from scipy.fft import fftn, ifftn, fftshift, ifftshift
import time


@profile()
def main():

    t0 = time.time()

    image = np.load(r"E:\Date\Teste\7_MDPI\Training image\Data.npy")
    l = np.uint16(image.shape[0] / 2)
    k = 250
    k = np.uint16(k / 2)
    inverse_nir = ifftn(fftshift(fftn(image[:, :, 0], norm="ortho"))[l - k:l + k, l - k:l + k], norm="ortho")
    inverse_red = ifftn(fftshift(fftn(image[:, :, 1], norm="ortho"))[l - k:l + k, l - k:l + k], norm="ortho")
    inverse_gre = ifftn(fftshift(fftn(image[:, :, 2], norm="ortho"))[l - k:l + k, l - k:l + k], norm="ortho")
    inverse_blu = ifftn(fftshift(fftn(image[:, :, 3], norm="ortho"))[l - k:l + k, l - k:l + k], norm="ortho")
    inverse_nir = np.expand_dims(inverse_nir / inverse_nir.max() * 255, axis=-1)
    inverse_red = np.expand_dims(inverse_red / inverse_red.max() * 255, axis=-1)
    inverse_gre = np.expand_dims(inverse_gre / inverse_gre.max() * 255, axis=-1)
    inverse_blu = np.expand_dims(inverse_blu / inverse_blu.max() * 255, axis=-1)
    inverse = np.abs(np.concatenate((inverse_nir, inverse_red, inverse_gre, inverse_blu), axis=-1))
    inverse[:, :, 0] = inverse[:, :, 0] + np.mean(image[:, :, 0]) - np.mean(inverse[:, :, 0])
    inverse[:, :, 1] = inverse[:, :, 1] + np.mean(image[:, :, 1]) - np.mean(inverse[:, :, 1])
    inverse[:, :, 2] = inverse[:, :, 2] + np.mean(image[:, :, 2]) - np.mean(inverse[:, :, 2])
    inverse[:, :, 3] = inverse[:, :, 3] + np.mean(image[:, :, 3]) - np.mean(inverse[:, :, 3])
    inverse = np.uint8(inverse)
    np.save(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DFT\Data_DFT_250.npy", inverse)
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
    print(np.flip(centers[:, 1:4], axis=-1))
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(inverse.shape)
    segmented_image = cv2.resize(segmented_image, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    segmented_image = segmented_image[:, :, 1:4]
    cv2.imwrite(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DFT\GMM_5_DFT_250.TIF", segmented_image)

    t5 = time.time()
    prediction_time_seconds = t5 - t4

    print("Processing time = ", processing_time_seconds)
    print("Training time = ", training_time_seconds)
    print("Prediction time = ", prediction_time_seconds)

    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    labels = labels.reshape((inverse.shape[0], inverse.shape[1]))
    np.save(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DFT\Labels_5_DFT_250.npy", labels)
    labels = cv2.resize(labels, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    np.save(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_DFT\Labels_5_DFT_250_resized.npy", labels)

    # image = image[:, :, 1:4]
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(image)
    # axarr[1].imshow(segmented_image)
    # plt.show()


if __name__ == "__main__":
    main()


