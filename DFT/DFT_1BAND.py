import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift

image = np.load(r"E:\Date\Teste\7_MDPI\Training image\Data.npy")
image_nir = image[:, :, 0]
image_dft = fftn(image_nir, norm="ortho")
image_dft = fftshift(image_dft)

rows, cols = image_dft.shape
x = np.arange(0, cols, 1)
y = np.arange(0, rows, 1)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.log(np.abs(image_dft) + 1e-5), cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Coeficienții DFT')
plt.show()
