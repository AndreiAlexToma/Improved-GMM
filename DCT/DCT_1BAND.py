import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dctn, idctn

image = np.load(r"E:\Date\Teste\7_MDPI\Training image\Data.npy")
image_nir = image[:, :, 0]
image_dct = dctn(image_nir, norm="ortho")

rows, cols = image_dct.shape
x = np.arange(0, cols, 1)
y = np.arange(0, rows, 1)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.log(np.abs(image_dct) + 1e-5), cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Coeficienții DCT')
plt.show()

image_dct_200 = image_dct[:200, :200]
image_dct_400 = image_dct[:400, :400]
image_dct_500 = image_dct[:500, :500]
image_dct_600 = image_dct[:600, :600]
image_dct_800 = image_dct[:800, :800]
inverse_200 = idctn(image_dct_200, norm="ortho")
inverse_400 = idctn(image_dct_400, norm="ortho")
inverse_500 = idctn(image_dct_500, norm="ortho")
inverse_600 = idctn(image_dct_600, norm="ortho")
inverse_800 = idctn(image_dct_800, norm="ortho")

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.imshow(inverse_200)
plt.show()



