import numpy as np
import os
import random
import cv2
from fiteness import *

class DNA:
    def __init__(self, targetImage, brushesList):  
        self.target_image = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
        self.brush = self.resize_brush(random.choice(brushesList), 0.1, 0.3)
        #self.color = self.change_color(self.brush, -200, 200)
        self.color, self.mask = self.change_brush_color()
        self.brush = self.color
        self.fitness = None
        self.xy_position = None

    def resize_brush(self, brush, minRange, maxRange):
        new_height = random.randint(int(brush.shape[0] * minRange), int(brush.shape[0] * maxRange))
        new_width = random.randint(int(brush.shape[1] * minRange), int(brush.shape[1] * maxRange))

        new_height = max(1, new_height)
        new_width = max(1, new_width)
        return cv2.resize(brush, (new_width, new_height))
    
    def change_color(self, brush, minRange, maxRange):
        brightness_shift = random.randint(minRange, maxRange)
        return np.clip(brush.astype(int) + brightness_shift, 0, 255).astype(np.uint8)
    
    def change_brush_color(self):
        # Generar un tono de gris aleatorio
        random_gray = random.randint(0, 255)
        colored_brush = np.full_like(self.brush, random_gray)
        self.color = colored_brush
        # Crear una máscara usando el color negro como fondo
        mask = cv2.threshold(self.brush, 1, 255, cv2.THRESH_BINARY)[1]
        return colored_brush, mask
    
    def generate_random_position(self, canvas_height, canvas_width):
        max_x = canvas_width - self.brush.shape[1]
        max_y = canvas_height - self.brush.shape[0]
        self.xy_position = (random.randint(0, max_x), random.randint(0, max_y))
    
    def calculate_fitness(self):
        self.fitness = calcularFitnes(self.color,self.xy_position,img)
    
    def mutate(self, canvas):
        mutation_type = random.choice(["brush", "position", "both"])

        if mutation_type in ['brush', 'both']:
            choices = [self.resize_brush(self.brush, 0.5, 1),
                    self.change_color(self.brush, -200, 200),
                    self.resize_brush(self.change_color(self.brush, -200, 200), 0.5, 1)]
            self.brush = random.choice(choices)

        if mutation_type in ['position', 'both']:
            displacement = (random.randint(-50, 50), random.randint(-50, 50))
            self.xy_position = (max(0, min(self.xy_position[0] + displacement[0], canvas.width - self.brush.shape[1])),
                                max(0, min(self.xy_position[1] + displacement[1], canvas.height - self.brush.shape[0])))

    def mutate2(self, canvas):
        mutation_type = random.choice(["brush", "position", "both"])
        new_color, _ = self.change_brush_color()

        if mutation_type in ['brush', 'both']:
            self.brush = random.choice([self.resize_brush(self.brush, 0.5, 1), new_color])

        if mutation_type in ['position', 'both']:
            displacement = (random.randint(-50, 50), random.randint(-50, 50))
            self.xy_position = (max(0, min(self.xy_position[0] + displacement[0], canvas.width - self.brush.shape[1])),
                                max(0, min(self.xy_position[1] + displacement[1], canvas.height - self.brush.shape[0])))

        # Asegurarse de actualizar también la máscara después de la mutación
        _, self.mask = self.change_brush_color()
        
    def generate_mask_from_brush(self, brush):
        # Crear una máscara del mismo tamaño que la brocha, utilizando el color negro como fondo
        mask = cv2.threshold(brush, 1, 255, cv2.THRESH_BINARY)[1]
        return mask


    def crossover(self, parent1, parent2, img, brushesList, canvas):
        child = DNA(img, brushesList)

        # Escoger de manera aleatoria la forma del pincel, si del parent1 o del parent2
        #child.brush = random.choice([self.resize_brush(parent1.brush, 0.1, 1.5), self.resize_brush(parent2.brush, 0.1, 1.5)])
#        child.brush = random.choice([parent1.brush, parent2.brush])

        # Redimensionar las imágenes de los padres para que tengan las mismas dimensiones
        common_height = min(parent1.color.shape[0], parent2.color.shape[0])
        common_width = min(parent1.color.shape[1], parent2.color.shape[1])

        parent1_color_resized = cv2.resize(parent1.color, (common_width, common_height))
        parent2_color_resized = cv2.resize(parent2.color, (common_width, common_height))

        # Aplicar crossover en el color del pincel
        alpha = random.uniform(0, 1)  # Factor de mezcla para el color
        child_color  = cv2.addWeighted(parent1_color_resized, alpha, parent2_color_resized, 1 - alpha, 0) #Combinar colores de los padres
        child.brush = child_color
        child.mask = self.generate_mask_from_brush(child_color)

        # Heredar la posición de uno de los padres con un desplazamiento aleatorio dentro del lienzo
        displacement = (random.randint(-70, 70), random.randint(-70, 70)) 
        if random.random() < 0.5: 
            child.xy_position = (max(0, min(parent1.xy_position[0] + displacement[0], canvas.width - child.brush.shape[1])),
                                max(0, min(parent1.xy_position[1] + displacement[1], canvas.height - child.brush.shape[0])))
        else:
            child.xy_position = (max(0, min(parent2.xy_position[0] + displacement[0], canvas.width - child.brush.shape[1])),
                                max(0, min(parent2.xy_position[1] + displacement[1], canvas.height - child.brush.shape[0])))

        return child

class Canvas:
    def __init__(self, img):
        self.height, self.width = img.shape[:2]
        self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)

    def paint(self, population):
        for individual in population:
            paste_x, paste_y = individual.xy_position 

            # Aplica la brocha usando la máscara para evitar el fondo
            brush_height, brush_width = individual.brush.shape[:2]
            mask_inv = cv2.bitwise_not(individual.mask)
            roi = self.canvas[paste_y:paste_y+brush_height, paste_x:paste_x+brush_width]
            individual.brush = cv2.bitwise_and(individual.brush, individual.brush, mask=individual.mask)
            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            dst = cv2.add(bg, individual.brush)

            # Actualiza el lienzo con la brocha aplicada
            self.canvas[paste_y:paste_y+brush_height, paste_x:paste_x+brush_width] = dst

        # Muestra el lienzo después de pintar toda la población
        cv2.imshow('Image', self.canvas)
        cv2.waitKey(0)

class GeneticAlgorithm:
    def __init__(self, targetImage, populationSize, maxGenerations, brushesList):
        self.target_image = targetImage
        self.population_size = populationSize
        self.max_generations = maxGenerations
        self.brushesList = brushesList
        self.current_population = []
        self.canvas = Canvas(targetImage)

    def select_parents(self, percentage): # Selección de los mejores padres para la evolución
        sorted_population = sorted(self.current_population, key=lambda x: x.fitness, reverse=True) #Obtener los mejores por fitness
        selected_parents = sorted_population[:int(len(self.current_population) * percentage)]
        return selected_parents

    def initialize_population(self): #Inicializa la primera población con características aleatorias
        for i in range(self.population_size):
            new_individual = DNA(self.target_image, self.brushesList)
            new_individual.generate_random_position(self.canvas.height, self.canvas.width)
            self.current_population.append(new_individual)
        self.canvas.paint(self.current_population)
        self.evolve() #Al finalizar la población inicial, se comienza con las generaciones (evolución)

    def evolve(self):
        for generation in range(self.max_generations):
            # Calcular el fitness de cada individuo en la población actual
            for individuo in self.current_population:
                individuo.calculate_fitness()

            # Selección de los mejores individuos de la generación actual
            best_individuals = self.select_parents(0.1)

            """print(f"Generación {generation + 1}:") # Borrarlo después
            for idx, individual in enumerate(best_individuals, start=1):
                print(f"  Mejor individuo {idx}: Fitness = {individual.fitness}")"""

            new_population = []
            for _ in range(random.randint(10,20)): #Aleatoriedad para la cantidad de individuos que se generarán para las próximas generaciones
                parent1 = random.choice(best_individuals)
                parent2 = random.choice(best_individuals)
                child = parent1.crossover(parent1, parent2, self.target_image, self.brushesList, self.canvas)
                child.resize_brush(child.brush, 0.5, 1.5)
                new_population.append(child)

                if random.random() < 0.1:
                    print("Hijo mutará") # Borrarlo después 
                    new_individual = DNA(self.target_image, self.brushesList)
                    new_individual.generate_random_position(self.canvas.height, self.canvas.width)
                    new_individual.resize_brush(new_individual.brush, 0.1, 0.3)
                    self.current_population.append(new_individual)
                    new_population.append(new_individual)
            self.current_population = new_population
            self.canvas.paint(self.current_population)
            

# Carga todos los brochazos en un array
def load_brushes():
    brushes = []
    for filename in os.listdir('Brushes/'):
        brush = cv2.imread('Brushes/' + filename, cv2.IMREAD_GRAYSCALE)
        #print(filename)
        brushes.append(brush)
    return brushes


def start(targetImage, populationSize, maxGenerations):
    brushes_list = load_brushes()

    img = cv2.imread(targetImage)

    startGeneticAlgorithm = GeneticAlgorithm(img , populationSize, maxGenerations, brushes_list)
    startGeneticAlgorithm.initialize_population()

img = "Images/PALETA.jpg"
populationSize = 30
maxGenerations = 1000
start(img, populationSize, maxGenerations)
