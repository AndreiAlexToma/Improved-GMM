import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from memory_profiler import profile
import time


@profile
def main():
    t0 = time.time()

    image = np.load(r"E:\Date\Teste\3_ISMSIT Turcia 2022\final_results\Input\test\4000\RGBN\2135_530.npy")
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
    # cv2.imwrite(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_Autoencoder\GMM_1200_001_10_2.TIF", segmented_image)

    t5 = time.time()
    prediction_time_seconds = t5 - t4

    print("Processing time = ", processing_time_seconds)
    print("Training time = ", training_time_seconds)
    print("Prediction time = ", prediction_time_seconds)

    labels = labels.reshape((image.shape[0], image.shape[1]))
    # np.save(r"E:\Date\Teste\7_MDPI\Training image\GMM_5_Autoencoder\GMM_1200_001_10_2.npy", labels)

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    image = image[:, :, 1:4]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image)
    axarr[1].imshow(segmented_image)
    plt.show()


if __name__ == "__main__":
    main()
