import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift

image = np.load(r"E:\Date\Teste\7_MDPI\Training image\Data.npy")
l = np.uint16(image.shape[0]/2)
k = 500
k = np.uint16(k/2)
image_dft_nir = fftshift(fftn(image[:, :, 0], norm="ortho"))
image_dft_red = fftshift(fftn(image[:, :, 1], norm="ortho"))
image_dft_gre = fftshift(fftn(image[:, :, 2], norm="ortho"))
image_dft_blu = fftshift(fftn(image[:, :, 3], norm="ortho"))

extracted_nir = image_dft_nir[l-k:l+k, l-k:l+k]
extracted_red = image_dft_red[l-k:l+k, l-k:l+k]
extracted_gre = image_dft_gre[l-k:l+k, l-k:l+k]
extracted_blu = image_dft_blu[l-k:l+k, l-k:l+k]

inverse_nir = np.expand_dims(ifftn(extracted_nir, norm="ortho"), axis=-1)
inverse_red = np.expand_dims(ifftn(extracted_red, norm="ortho"), axis=-1)
inverse_gre = np.expand_dims(ifftn(extracted_gre, norm="ortho"), axis=-1)
inverse_blu = np.expand_dims(ifftn(extracted_blu, norm="ortho"), axis=-1)
inverse = np.concatenate((inverse_nir, inverse_red, inverse_gre, inverse_blu), axis=-1)
inverse = np.abs(inverse)
# inverse[:, :, 0] = inverse[:, :, 0] / inverse[:, :, 0].max() * 255
# inverse[:, :, 1] = inverse[:, :, 1] / inverse[:, :, 1].max() * 255
# inverse[:, :, 2] = inverse[:, :, 2] / inverse[:, :, 2].max() * 255
# inverse[:, :, 3] = inverse[:, :, 3] / inverse[:, :, 3].max() * 255

image = image[:, :, 1:4]
image = np.flip(image, axis=-1)

inverse = inverse[:, :, 1:4]
inverse = np.flip(inverse, axis=-1)

f, axarr = plt.subplots(1, 3)
axarr[0].imshow(image)
axarr[1].imshow(np.uint8(inverse))

inverse[:, :, 0] = inverse[:, :, 0] / inverse[:, :, 0].max() * 255
inverse[:, :, 1] = inverse[:, :, 1] / inverse[:, :, 1].max() * 255
inverse[:, :, 2] = inverse[:, :, 2] / inverse[:, :, 2].max() * 255

axarr[2].imshow(np.uint8(inverse))
plt.show()

