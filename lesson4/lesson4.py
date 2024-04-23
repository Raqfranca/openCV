import os
import cv2

#Resize to image
# weight and height
img = cv2.imread(os.path.join('.', 'data', 'dog.jpg'))

resized_img = cv2.resize(img, (640,480))

#print(resized_img.shape)
print(img.shape)

cv2.imshow('img', img)
#cv2.imshow('resized_img', resized_img)

#crop to image
""" 
Para criar uma imagem centralizada:

altura_desejada = 825 // 2  # Metade da altura original
largura_desejada = 1100 // 2  # Metade da largura original

x_top_left = (1100 - largura_desejada) // 2
y_top_left = (825 - altura_desejada) // 2

x_bottom_right = x_top_left + largura_desejada
y_bottom_right = y_top_left + altura_desejada

cropped_img = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]

x_top_left = 2755
y_top_left = 206,25

x_bottom_right = 825
y_bottom_right = 618,75 

O primeiro conjunto de números (100:500) representa as coordenadas da altura (eixo y), 
onde 100 é o início da região a ser cortada e 500 é o final.

O segundo conjunto de números (290:790) representa as coordenadas da largura (eixo x), 
onde 290 é o início da região a ser cortada e 790 é o final.

"""

cropped_img = img[100:500, 290:790]

print(cropped_img.shape)

cv2.imshow('cropped_img', cropped_img)
cv2.waitKey(0)

