# Muhamad Putra Satria
# F55120009
# A - Teknik Informatika

import numpy as np
import matplotlib.pyplot as plt
import cv2

gambar = cv2.imread("Gambar/Reid_Hoffman.jpg")
gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
pixel_values = gambar.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
kriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, kriteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
labels = labels.flatten()
hasil_segmentasi = centers[labels.flatten()]
hasil_segmentasi = hasil_segmentasi.reshape(gambar.shape)

plt.imshow(hasil_segmentasi)
plt.show()










