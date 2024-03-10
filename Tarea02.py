import numpy as np
import os
import random
import cv2

class DNA:
    def __init__(self, targetImage, brushesList):  
        self.target_image = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
        self.brush = self.resize_brush(random.choice(brushesList), 0.1, 0.3)
        self.color = self.change_color(self.brush, -100, 100)
        self.brush = self.color
        self.fitness = None


    def resize_brush(self, brush, minRange, maxRange):
        new_height = random.randint(int(brush.shape[0] * minRange), int(brush.shape[0] * maxRange))
        new_width = random.randint(int(brush.shape[1] * minRange), int(brush.shape[1] * maxRange))
        return cv2.resize(brush, (new_width, new_height))
    
    def change_color(self, brush, minRange, maxRange):
        brightness_shift = random.randint(minRange, maxRange)
        return np.clip(brush.astype(int) + brightness_shift, 0, 255).astype(np.uint8)
    
    def calculate_fitness(self):
        self.fitness = random.randint(0,100)
    
    def mutate(self):
        choices = [self.resize_brush(self.brush, 0.5, 0.8),
           self.change_color(self.brush, -100, 100),
           self.resize_brush(self.change_color(self.brush, -100, 100), 0.3, 0.8)]
        self.brush = random.choice(choices)

    def crossover(self, parent1, parent2, img, brushesList):
        child = DNA(img, brushesList)

        # Escoger de manera aleatoria la forma del pincel, si del parent1 o del parent2
        child.brush = random.choice([self.resize_brush(parent1.brush, 0.5, 1), self.resize_brush(parent2.brush, 0.5, 1)])

        # Redimensionar las imágenes de los padres para que tengan las mismas dimensiones
        common_height = min(parent1.color.shape[0], parent2.color.shape[0])
        common_width = min(parent1.color.shape[1], parent2.color.shape[1])

        parent1_color_resized = cv2.resize(parent1.color, (common_width, common_height))
        parent2_color_resized = cv2.resize(parent2.color, (common_width, common_height))

        # Aplicar crossover en el color del pincel
        alpha = random.uniform(0, 1)  # Factor de mezcla para el color
        child.color = cv2.addWeighted(parent1_color_resized, alpha, parent2_color_resized, 1 - alpha, 0) #Combinar colores de los padres

        return child

class Canvas:
    def __init__(self, img):
        self.height, self.width = img.shape[:2]
        self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)

    def paint(self, population):
        for individual in population:
            max_x = self.canvas.shape[1] - individual.brush.shape[1]
            max_y = self.canvas.shape[0] - individual.brush.shape[0]
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # Pinta en la posicion definida 
            self.canvas[paste_y:paste_y + individual.brush.shape[0], paste_x:paste_x + individual.brush.shape[1]] = individual.brush

            cv2.imshow('Image', self.canvas)
            cv2.waitKey(500)
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
            self.current_population.append(new_individual)
        self.canvas.paint(self.current_population)
        self.evolve() #Al finalizar la población inicial, se comienza con las generaciones (evolución)

    def evolve(self):
        for generation in range(self.max_generations):
            # Calcular el fitness de cada individuo en la población actual
            for individuo in self.current_population:
                individuo.calculate_fitness()

            # Selección de los mejores individuos de la generación actual
            best_individuals = self.select_parents(0.5)

            print(f"Generación {generation + 1}:") # Borrarlo después
            for idx, individual in enumerate(best_individuals, start=1):
                print(f"  Mejor individuo {idx}: Fitness = {individual.fitness}")

            new_population = []
            for _ in range(random.randint(10,20)): #Aleatoriedad para la cantidad de individuos que se generarán para las próximas generaciones
                parent1 = random.choice(best_individuals)
                parent2 = random.choice(best_individuals)
                child = parent1.crossover(parent1, parent2, self.target_image, self.brushesList)
                
                if random.random() < 0.3:
                    print("Hijo mutará") # Borrarlo después
                    child.mutate() 
                new_population.append(child)
            self.current_population = new_population
            self.canvas.paint(self.current_population)
            

# Carga todos los brochazos en un array
def load_brushes():
    brushes = []
    for filename in os.listdir('Brushes/'):
        brush = cv2.imread('Brushes/' + filename, cv2.IMREAD_GRAYSCALE)
        print(filename)
        brushes.append(brush)
    return brushes


def start(targetImage, populationSize, maxGenerations):
    brushes_list = load_brushes()

    img = cv2.imread(targetImage)

    startGeneticAlgorithm = GeneticAlgorithm(img , populationSize, maxGenerations, brushes_list)
    startGeneticAlgorithm.initialize_population()

img = "Images/pokemon.jpg"
start(img, 10,1)
