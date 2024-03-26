import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2

image = np.load(r"E:\Date\Teste\7_MDPI\Training image\Data.npy")
image_nir = image[:, :, 0]
image_red = image[:, :, 1]
image_gre = image[:, :, 2]
image_blu = image[:, :, 3]

wavelet = 'haar'

# Apply DWT to each channel
nir_coeffs = pywt.wavedec2(image_nir, wavelet, level=1)
red_coeffs = pywt.wavedec2(image_red, wavelet, level=1)
gre_coeffs = pywt.wavedec2(image_gre, wavelet, level=1)
blu_coeffs = pywt.wavedec2(image_blu, wavelet, level=1)

# Display the original RGB image
plt.subplot(2, 2, 1)
plt.imshow(np.flip(image[:, :, 1:4], axis=-1))
plt.title('Original NRGB Image')

# Display the approximation coefficients for each channel 1st lvl WITHOUT BRIGHTNESS TRANSFER AND NORM
plt.subplot(2, 2, 2)
red_coeffs_aprox = np.expand_dims(red_coeffs[0], axis=-1)
gre_coeffs_aprox = np.expand_dims(gre_coeffs[0], axis=-1)
blu_coeffs_aprox = np.expand_dims(blu_coeffs[0], axis=-1)
nir_coeffs_aprox = np.expand_dims(nir_coeffs[0], axis=-1)
aprox_rgb = np.concatenate((blu_coeffs_aprox, gre_coeffs_aprox, red_coeffs_aprox), axis=-1)
# aprox_rgb[:, :, 0] = aprox_rgb[:, :, 0] * 255 / (np.max(aprox_rgb[:, :, 0]))
# aprox_rgb[:, :, 1] = aprox_rgb[:, :, 1] * 255 / (np.max(aprox_rgb[:, :, 1]))
# aprox_rgb[:, :, 2] = aprox_rgb[:, :, 2] * 255 / (np.max(aprox_rgb[:, :, 2]))
plt.imshow(np.uint8(aprox_rgb))
plt.title('Approximation Coefficients 1st Level')

# Display the approximation coefficients for each channel 2nd lvl WITHOUT BRIGHTNESS TRANSFER AND NORM
plt.subplot(2, 2, 3)
nir_coeffs_2 = pywt.wavedec2(nir_coeffs[0], wavelet, level=1)
red_coeffs_2 = pywt.wavedec2(red_coeffs[0], wavelet, level=1)
gre_coeffs_2 = pywt.wavedec2(gre_coeffs[0], wavelet, level=1)
blu_coeffs_2 = pywt.wavedec2(blu_coeffs[0], wavelet, level=1)

nir_coeffs_2_aprox = np.expand_dims(nir_coeffs_2[0], axis=-1)
red_coeffs_2_aprox = np.expand_dims(red_coeffs_2[0], axis=-1)
gre_coeffs_2_aprox = np.expand_dims(gre_coeffs_2[0], axis=-1)
blu_coeffs_2_aprox = np.expand_dims(blu_coeffs_2[0], axis=-1)

aprox_2_rgb = np.concatenate((blu_coeffs_2_aprox, gre_coeffs_2_aprox, red_coeffs_2_aprox), axis=-1)
# aprox_2_rgb[:, :, 0] = aprox_2_rgb[:, :, 0] * 255 / (np.max(aprox_2_rgb[:, :, 0]))
# aprox_2_rgb[:, :, 1] = aprox_2_rgb[:, :, 1] * 255 / (np.max(aprox_2_rgb[:, :, 1]))
# aprox_2_rgb[:, :, 2] = aprox_2_rgb[:, :, 2] * 255 / (np.max(aprox_2_rgb[:, :, 2]))

plt.imshow(np.uint8(aprox_2_rgb))
plt.title('Approximation Coefficients 2nd Level')

# Display the approximation coefficients for each channel 3rd lvl WITHOUT BRIGHTNESS TRANSFER AND NORM
plt.subplot(2, 2, 4)
nir_coeffs_3 = pywt.wavedec2(nir_coeffs_2[0], wavelet, level=1)
red_coeffs_3 = pywt.wavedec2(red_coeffs_2[0], wavelet, level=1)
gre_coeffs_3 = pywt.wavedec2(gre_coeffs_2[0], wavelet, level=1)
blu_coeffs_3 = pywt.wavedec2(blu_coeffs_2[0], wavelet, level=1)

nir_coeffs_3_aprox = np.expand_dims(nir_coeffs_3[0], axis=-1)
red_coeffs_3_aprox = np.expand_dims(red_coeffs_3[0], axis=-1)
gre_coeffs_3_aprox = np.expand_dims(gre_coeffs_3[0], axis=-1)
blu_coeffs_3_aprox = np.expand_dims(blu_coeffs_3[0], axis=-1)

aprox_3_rgb = np.concatenate((blu_coeffs_3_aprox, gre_coeffs_3_aprox, red_coeffs_3_aprox), axis=-1)
# aprox_3_rgb[:, :, 0] = aprox_3_rgb[:, :, 0] * 255 / (np.max(aprox_3_rgb[:, :, 0]))
# aprox_3_rgb[:, :, 1] = aprox_3_rgb[:, :, 1] * 255 / (np.max(aprox_3_rgb[:, :, 1]))
# aprox_3_rgb[:, :, 2] = aprox_3_rgb[:, :, 2] * 255 / (np.max(aprox_3_rgb[:, :, 2]))

plt.imshow(np.uint8(aprox_3_rgb))
plt.title('Approximation Coefficients 3rd Level')

plt.show()