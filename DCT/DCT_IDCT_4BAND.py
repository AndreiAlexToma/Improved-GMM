import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn

image = np.load(r"E:\Date\Teste\7_AECE\Training image\Data.npy")

k = 500

image_dct_nir = dctn(image[:, :, 0], norm="ortho")
image_dct_red = dctn(image[:, :, 1], norm="ortho")
image_dct_gre = dctn(image[:, :, 2], norm="ortho")
image_dct_blu = dctn(image[:, :, 3], norm="ortho")

extracted_nir = image_dct_nir[:k, :k]
extracted_red = image_dct_red[:k, :k]
extracted_gre = image_dct_gre[:k, :k]
extracted_blu = image_dct_blu[:k, :k]

inverse_nir = np.expand_dims(idctn(extracted_nir, norm="ortho"), axis=-1)
inverse_red = np.expand_dims(idctn(extracted_red, norm="ortho"), axis=-1)
inverse_gre = np.expand_dims(idctn(extracted_gre, norm="ortho"), axis=-1)
inverse_blu = np.expand_dims(idctn(extracted_blu, norm="ortho"), axis=-1)
inverse = np.concatenate((inverse_nir, inverse_red, inverse_gre, inverse_blu), axis=-1)
# inverse = cv2.resize(inverse, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
# inverse = np.flip(inverse[:, :, 1:4], axis=-1)
# inverse = np.uint8(inverse)
print(inverse.shape)

normalized_nir = inverse_nir / inverse_nir.max() * 255
normalized_red = inverse_red / inverse_red.max() * 255
normalized_gre = inverse_gre / inverse_gre.max() * 255
normalized_blu = inverse_blu / inverse_blu.max() * 255

inverse = np.concatenate((normalized_nir, normalized_red, normalized_gre, normalized_blu), axis=-1)

brightness_image_nir = np.mean(image[:, :, 0])
brightness_image_red = np.mean(image[:, :, 1])
brightness_image_gre = np.mean(image[:, :, 2])
brightness_image_blu = np.mean(image[:, :, 3])
brightness_inverse_nir = np.mean(inverse[:, :, 0])
brightness_inverse_red = np.mean(inverse[:, :, 1])
brightness_inverse_gre = np.mean(inverse[:, :, 2])
brightness_inverse_blu = np.mean(inverse[:, :, 3])

brightness_difference_nir = brightness_image_nir - brightness_inverse_nir
brightness_difference_red = brightness_image_red - brightness_inverse_red
brightness_difference_gre = brightness_image_gre - brightness_inverse_gre
brightness_difference_blu = brightness_image_blu - brightness_inverse_blu

inverse[:, :, 0] = inverse[:, :, 0] + brightness_difference_nir
inverse[:, :, 1] = inverse[:, :, 1] + brightness_difference_red
inverse[:, :, 2] = inverse[:, :, 2] + brightness_difference_gre
inverse[:, :, 3] = inverse[:, :, 3] + brightness_difference_blu
adjusted_image = inverse
image = image[:, :, 1:4]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
inverse = inverse[:, :, 1:4]
inverse = np.uint8(inverse)
inverse = np.flip(inverse, axis=-1)


f, axarr = plt.subplots(1, 2)
axarr[0].imshow(image)
axarr[1].imshow(inverse)
plt.show()

