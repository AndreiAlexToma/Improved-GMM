import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2

image = np.load(r"E:\Date\Teste\7_MDPI\Data\Full\4136_710_0.npy")
img_array = image[:, :, 0]

# Set wavelet and decomposition level
wavelet = 'haar'

# Perform 2D Discrete Wavelet Transform
coeffs = pywt.wavedec2(img_array, wavelet, level=1)

# Display the original image
plt.subplot(1, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')

# Display the approximation coefficients
plt.subplot(1, 3, 2)
plt.imshow(coeffs[0], cmap='gray')
plt.title('Approximation Coefficients')

# Display the detail coefficients
plt.subplot(1, 3, 3)
plt.imshow(coeffs[1][1], cmap='gray')
print(coeffs[1][0])
plt.title(f'Horizontal Detail Coefficients')

plt.show()