import numpy as np
import cv2

class Individuo:
    def __init__(self, imagen_objetivo): #Mandar el arreglo de brochas
        self.imagen_objetivo = cv2.cvtColor(imagen_objetivo, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises
        self.tamano = np.random.uniform(0.1, 0.3)  
        self.color = np.random.randint(-100,100)  #valores aleatorios entre 0 y 255 para el R, G, B
        
        self.tipo_brocha = np.random.randint(1,4) #Choice para arreglo de imagenes Brushes/brush1.jpg, etc

        #Posición del individuo dentro de la imagen  falta considerar el tamaño de la brocha para que no se salga de la imagen original
        # self.tamano aplicado ya al tipo de brocha
        #ejemplo de como se aplica en  canvas[paste_y:paste_y+brush.shape[0], paste_x:paste_x+brush.shape[1]] = brush
        #self.posicion = (np.random.randint(0, imagen_objetivo.shape[1]), np.random.randint(0, imagen_objetivo.shape[0])) 
        self.fitness = None

    def imprimir_caracteristicas(self):
        print(f"Tamaño: {self.tamano}")
        print(f"Color: {self.color}")
        #print(f"Posición: {self.posicion}")
        print(f"Tipo de Brocha: {self.tipo_brocha}")
        print()
    
    def calcularFitness(self):
        print("prueba")

    def mutar(self): 
        print("prueba")
    
    def cruzar(self, individuo2):
        print("prueba")

class AlgoritmoGenetico:
    #Declaración de parámetros
    imagen_objetivo = "Imagenes/prueba1.png"
    tamano_poblacion = 10
    max_generaciones = 3
    poblacion_actual = []

    def inicializar_poblacion(self):
        for i in range(self.tamano_poblacion):
            nuevo_individuo = Individuo(self.imagen_objetivo)
            self.poblacion_actual.append(nuevo_individuo)
            print(f"Individuo {i + 1} - Características:")
            nuevo_individuo.imprimir_caracteristicas()

    """def evolucionar(self):
        for generacion in range(self.max_generaciones):
            # Calcular el fitness de cada individuo en la población actual
            for individuo in self.poblacion_actual:
                individuo.calcular_fitness()

            # Seleccionar individuos para la reproducción (puedes implementar diferentes estrategias de selección)
            padres_seleccionados = self.seleccionar_padres()

            # Crear descendencia mediante cruza y mutación
            descendencia = self.crear_descendencia(padres_seleccionados)

            # Reemplazar la población actual con la nueva generación (descendencia)
            self.poblacion_actual = descendencia

            # Realizar otras operaciones, como mostrar estadísticas, guardar mejores individuos, etc.
            #Mandar la lista de población actual a que se dibuje en el canvas. 

    def seleccionar_padres(self):
        # Implementar el método de selección de padres
        # Puedes usar métodos como la ruleta, torneo, etc.
        pass

    def crear_descendencia(self, padres):
        descendencia = []

        # Implementar la cruza y mutación para generar la descendencia
        # Puedes utilizar métodos como cruce en un punto, cruce uniforme, mutación por bit, etc.

        return descendencia"""
    
poblacion_inicial = AlgoritmoGenetico()
poblacion_inicial.inicializar_poblacion()