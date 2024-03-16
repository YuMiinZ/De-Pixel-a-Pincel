import numpy as np
from PIL import Image

def calcularFitness(matrizBrush, posicionBrush, section_image):
    def comparePixel(pixelBrush, pixelImage):
        pixelRate = abs(abs(pixelBrush - pixelImage) - 255)
        return pixelRate

    def aplicar_comparePixel(matrizImg, pixelBrush):
        # Obtener dimensiones de la matriz
        filas = len(matrizImg)
        columnas = len(matrizImg[0]) if filas > 0 else 0
        fitenessPixel = []
        # Recorrer la matriz y aplicar comparePixel a cada elemento
        for i in range(filas):
            for j in range(columnas):
                fitenessPixel.append(comparePixel(pixelBrush, matrizImg[i][j]))

        # Calcular la suma de los elementos
        promedio = sum(fitenessPixel) // len(fitenessPixel)
        return promedio

    def extractColor(matriz):
        conteo = {}
        for fila in matriz:
            for num in fila:
                if num in conteo:
                    conteo[num] += 1
                else:
                    conteo[num] = 1

        if conteo:  # Verificar si el diccionario no está vacío
            extractColor = None
            max_repeticiones = 0

            for num, repeticiones in conteo.items():
                if repeticiones > max_repeticiones:
                    extractColor = num
                    max_repeticiones = repeticiones

            return extractColor
        else:
            return 0  # Devolver un valor predeterminado si la matriz está vacía

    colorBrush = extractColor(matrizBrush)
    sizeBrush = np.array(matrizBrush).shape

    cuadro = section_image

    pixeles = aplicar_comparePixel(cuadro, colorBrush)
    return pixeles
