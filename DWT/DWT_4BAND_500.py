import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2

image = np.load(r"E:\Date\Teste\7_AECE\Training image\Data.npy")
image_nir = image[:, :, 0]
image_red = image[:, :, 1]
image_gre = image[:, :, 2]
image_blu = image[:, :, 3]

wavelet = 'haar'
level = 1

# Apply DWT to each channel
nir_coeffs = pywt.wavedec2(image_nir, wavelet, level=level)
red_coeffs = pywt.wavedec2(image_red, wavelet, level=level)
gre_coeffs = pywt.wavedec2(image_gre, wavelet, level=level)
blu_coeffs = pywt.wavedec2(image_blu, wavelet, level=level)

# Display the approximation coefficients for each channel WITHOUT BRIGHTNESS TRANSFER AND NORM
plt.subplot(2, 2, 1)
red_coeffs_aprox = np.expand_dims(red_coeffs[0], axis=-1)
gre_coeffs_aprox = np.expand_dims(gre_coeffs[0], axis=-1)
blu_coeffs_aprox = np.expand_dims(blu_coeffs[0], axis=-1)
nir_coeffs_aprox = np.expand_dims(nir_coeffs[0], axis=-1)
aprox_rgb = np.concatenate((blu_coeffs_aprox, gre_coeffs_aprox, red_coeffs_aprox), axis=-1)
# aprox_rgb[:, :, 0] = aprox_rgb[:, :, 0] * 255 / (np.max(aprox_rgb[:, :, 0]))
# aprox_rgb[:, :, 1] = aprox_rgb[:, :, 1] * 255 / (np.max(aprox_rgb[:, :, 1]))
# aprox_rgb[:, :, 2] = aprox_rgb[:, :, 2] * 255 / (np.max(aprox_rgb[:, :, 2]))
plt.imshow(np.uint8(aprox_rgb))
plt.title('Approximation Coefficients')

# Display the H detail coefficients for each channel
plt.subplot(2, 2, 2)
nir_coeffs_h = np.expand_dims(nir_coeffs[1][0], axis=-1)
red_coeffs_h = np.expand_dims(red_coeffs[1][0], axis=-1)
gre_coeffs_h = np.expand_dims(gre_coeffs[1][0], axis=-1)
blu_coeffs_h = np.expand_dims(blu_coeffs[1][0], axis=-1)
h_rgb = np.concatenate((blu_coeffs_h, gre_coeffs_h, red_coeffs_h), axis=-1)
h_rgb = ((h_rgb - np.min(h_rgb)) * 255 / (np.max(h_rgb) - np.min(h_rgb))) / 255
plt.imshow(h_rgb)
plt.title(f'H Detail Coefficients ')

# Display the V detail coefficients for each channel
plt.subplot(2, 2, 3)
nir_coeffs_v = np.expand_dims(nir_coeffs[1][1], axis=-1)
red_coeffs_v = np.expand_dims(red_coeffs[1][1], axis=-1)
gre_coeffs_v = np.expand_dims(gre_coeffs[1][1], axis=-1)
blu_coeffs_v = np.expand_dims(blu_coeffs[1][1], axis=-1)
v_rgb = np.concatenate((blu_coeffs_v, gre_coeffs_v, red_coeffs_v), axis=-1)
v_rgb = ((v_rgb - np.min(v_rgb)) * 255 / (np.max(v_rgb) - np.min(v_rgb))) / 255
plt.imshow(v_rgb)
plt.title(f'V Detail Coefficients ')

# Display the D detail coefficients for each channel
plt.subplot(2, 2, 4)
nir_coeffs_d = np.expand_dims(nir_coeffs[1][2], axis=-1)
red_coeffs_d = np.expand_dims(red_coeffs[1][2], axis=-1)
gre_coeffs_d = np.expand_dims(gre_coeffs[1][2], axis=-1)
blu_coeffs_d = np.expand_dims(blu_coeffs[1][2], axis=-1)
d_rgb = np.concatenate((blu_coeffs_d, gre_coeffs_d, red_coeffs_d), axis=-1)
d_rgb = ((d_rgb - np.min(d_rgb)) * 255 / (np.max(d_rgb) - np.min(d_rgb))) / 255
plt.imshow(d_rgb)
plt.title(f'D Detail Coefficients ')
plt.show()
