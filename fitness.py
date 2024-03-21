from __future__ import division   # impone aritmética no entera en la división
from PIL import Image             # funciones para cargar y manipular imágenes
import numpy as np                # funciones numéricas (arrays, matrices, etc.)
import matplotlib.pyplot as plt 


def comparePixel(pixelBrush,pixelImage):
    pixelRate = abs(abs(pixelBrush-pixelImage)-255)
    return pixelRate

def aplicar_comparePixel(matrizImg,pixelBrush):
    # Obtener dimensiones de la matriz
    filas = len(matrizImg)
    columnas = len(matrizImg[0]) if filas > 0 else 0
    fitenessPixel = []
    # Recorrer la matriz y aplicar comparePixel a cada elemento
    for i in range(filas):
        for j in range(columnas):
            fitenessPixel.append(comparePixel(pixelBrush,matrizImg[i][j]))
    
    # Calcular la suma de los elementos
    promedio = sum(fitenessPixel) // len(fitenessPixel)
    #print(promedio)

    return promedio


def extractColor(matriz):
    conteo = {}
    for fila in matriz:
        for num in fila:
            if num in conteo:
                conteo[num] += 1
            else:
                conteo[num] = 1
    
    extractColor = None
    max_repeticiones = 0
    
    for num, repeticiones in conteo.items():
        if repeticiones > max_repeticiones:
            extractColor = num
            max_repeticiones = repeticiones
    
    return extractColor


def calcularFitness(matrizBrush,posicionBrush,img):#posicionBrush con coordenadas x,y
    #print("color:------------------------------------------------------------------------------------")
    colorBrush = extractColor(matrizBrush)
    #print(colorBrush)
    #print("pos:")
    #print(posicionBrush)
    #print("tam:")
    sizeBrush = np.array(matrizBrush).shape # y , x
    #print(sizeBrush)

    I = Image.open(img)
    I = I.convert('L')
    a = np.asarray(I,dtype=np.int16)
    #sizeBrush = [40,80] # y, x
    posBrush = [posicionBrush[1],posicionBrush[0]] # y,x
    cuadro = a[posBrush[0]:posBrush[0]+sizeBrush[0],posBrush[1]:posBrush[1]+sizeBrush[1]]


    pixeles = aplicar_comparePixel(cuadro,colorBrush)
    #print("fitnes total:")
    #print(pixeles)
    #plt.imshow(np.asarray(cuadro),cmap='gray',interpolation='nearest')
    #plt.show()
    #print(pixeles)

    #print (I.size, I.mode, I.format)


    return pixeles
    
