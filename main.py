import numpy as np
import os
import random
import cv2

# Carga todos los brochazos en un array
def load_brushes():
    brushes = []
    for filename in os.listdir('Brushes/'):
        brush = cv2.imread('Brushes/' + filename, cv2.IMREAD_GRAYSCALE)
        brushes.append(brush)
    return brushes

# Carga la imagen a copiar y crea el canvas a pintar
def load_image_create_canvas(source):
    img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)

    # Crea el canvas 
    height, width = img.shape[:2]
    canvas = np.zeros((height, width), dtype=np.uint8)
    # Se selecciona una imagen para pintar
    # Esto seria por X cantidad de veces hasta que se vean igual
    brushes = load_brushes()
    for i in range(10):
        brush = resize_brush(random.choice(brushes))
        brush = change_color(brush)
        paint(canvas, brush)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_brush(brush):
    new_height = random.randint(int(brush.shape[0] * 0.3), int(brush.shape[0] * 0.3))
    new_width = random.randint(int(brush.shape[1] * 0.3), int(brush.shape[1] * 0.3))
    return cv2.resize(brush, (new_width, new_height))

def change_color(brush):
    # Apply random brightness shift to each pixel
    brightness_shift = random.randint(-100, 100)
    return np.clip(brush.astype(int) + brightness_shift, 0, 255).astype(np.uint8)



# Pinta el brochazo en el canvas
def paint(canvas, brush):
    # Define la posicion random del x y y 
    max_x = canvas.shape[1] - brush.shape[1]
    max_y = canvas.shape[0] - brush.shape[0]
    paste_x = random.randint(0, max_x)
    paste_y = random.randint(0, max_y)

    # Pinta en la posicion definida 
    canvas[paste_y:paste_y+brush.shape[0], paste_x:paste_x+brush.shape[1]] = brush
    cv2.imshow('Image', canvas)
    cv2.waitKey(0)

load_image_create_canvas('Images/pokemon.jpg')
