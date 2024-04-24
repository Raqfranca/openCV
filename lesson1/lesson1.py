import cv2
import os

# read image

image_path = os.path.join('.', 'data', 'bird.jpg')

img = cv2.imread(image_path)

# write image

cv2.imwrite(os.path.join('.', 'data', 'bird_out.jpg'), img)

# visualize image

cv2.imshow('imagem', img)
cv2.waitKey(0)
