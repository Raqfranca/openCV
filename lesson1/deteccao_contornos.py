import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('image.jpg')

# Converter a imagem para escala de cinza
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar um filtro de suavização (opcional)
imagem_blur = cv2.GaussianBlur(imagem_gray, (5, 5), 0)

# Detectar as bordas na imagem usando Canny
bordas = cv2.Canny(imagem_blur, 30, 150)

# Encontrar os contornos na imagem
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenhar os contornos na imagem original
cv2.drawContours(imagem, contornos, -1, (0, 255, 0), 3)

# Exibir a imagem resultante
cv2.imshow('Imagem com Contornos Detectados', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
