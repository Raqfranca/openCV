import cv2
import os

# read image

image_path = os.path.join('.', 'images', 'messi.jpg')

img = cv2.imread(image_path)

# visualize image

cv2.imshow('img', img)
cv2.waitKey(0)