from PIL import Image
import numpy as np
import os
import random
import cv2


def load_brushes():
    brushes = []
    for filename in os.listdir('Brushes/'):
        brush = cv2.imread('Brushes/' + filename, cv2.IMREAD_GRAYSCALE)
        brushes.append(brush)
    return brushes


def extract_color(matrix):
    pass


def calculate_similarity(section1, section2):
    # Asegurarse de que ambas secciones tengan el mismo tamaño
    if section1.shape != section2.shape:
        raise ValueError("Las secciones de imagen deben tener el mismo tamaño")

    # Calcular la diferencia cuadrática media (MSE) entre las secciones de imagen
    mse = np.mean((section1 - section2) ** 2)
    similarity = 1 / (1 + mse)  # Mayor similitud para valores de MSE más bajos
    return similarity


def divide_image_into_sections(image, num_sections):
    sections = []
    height, width = image.shape[:2]
    section_height = height // num_sections
    section_width = width // num_sections

    for i in range(num_sections):
        for j in range(num_sections):
            start_row = i * section_height
            end_row = start_row + section_height
            start_col = j * section_width
            end_col = start_col + section_width

            section = image[start_row:end_row, start_col:end_col]
            sections.append(section)

    return sections


class DNA:
    def __init__(self, target_image, brushes_list, canvas):
        self.target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        self.brush = self.resize_brush(random.choice(brushes_list), 0.1, 0.3)
        self.brush, self.mask = self.change_brush_color()
        self.fitness = None
        self.xy_position = None
        self.canvas = canvas  # Se guarda una referencia al lienzo
        print("Tamaño de la brocha:", self.brush.shape)

    def resize_brush(self, brush, minRange, maxRange):
        new_height = random.randint(int(brush.shape[0] * minRange), int(brush.shape[0] * maxRange))
        new_width = random.randint(int(brush.shape[1] * minRange), int(brush.shape[1] * maxRange))

        new_height = max(1, new_height)
        new_width = max(1, new_width)
        return cv2.resize(brush, (new_width, new_height))

    def change_brush_color(self):
        # Generar un tono de gris aleatorio
        random_gray = random.randint(0, 255)
        colored_brush = np.full_like(self.brush, random_gray)
        # Crear una máscara usando el color negro como fondo
        mask = cv2.threshold(self.brush, 1, 255, cv2.THRESH_BINARY)[1]
        return colored_brush, mask

    def generate_random_position(self, canvas_height, canvas_width):
        self.xy_position = (random.randint(0, canvas_width - self.brush.shape[1]),
                            random.randint(0, canvas_height - self.brush.shape[0]))

    def calculate_fitness(self):
        total_similarity = 0
        num_sections = self.canvas.num_sections
        for section in divide_image_into_sections(self.target_image, num_sections):
            section_x, section_y = self.xy_position
            canvas_section = self.canvas.canvas[section_y:section_y + section.shape[0],
                                                section_x:section_x + section.shape[1]]
            similarity = calculate_similarity(canvas_section, section)
            total_similarity += similarity

        self.fitness = total_similarity / num_sections ** 2

    def mutate(self, canvas):
        # Probabilidad de mutación del 3%
        mutation_probability = 0.03

        # Verificar si ocurre la mutación
        if random.random() < mutation_probability:
            self.generate_random_position(canvas.height, canvas.width)

    def crossover(self, parent1, parent2, brushes_list):
    # Escoge un punto de cruce aleatorio
        crossover_point = random.randint(0, min(len(parent1.brush), len(parent2.brush)))

        # Obtener las partes de los padres con las mismas dimensiones
        parent1_part = parent1.brush[:crossover_point]
        parent2_part = parent2.brush[:crossover_point]

        # Redimensionar las partes si es necesario
        common_height = min(parent1_part.shape[0], parent2_part.shape[0])
        common_width = min(parent1_part.shape[1], parent2_part.shape[1])
        parent1_part = parent1_part[:common_height, :common_width]
        parent2_part = parent2_part[:common_height, :common_width]

        # Combina las partes de los padres para crear un solo hijo
        child_brush = np.concatenate((parent1_part, parent2_part), axis=0)

        # Crea el objeto DNA para el hijo y devuelve una lista con él
        child = DNA(self.target_image, brushes_list, self.canvas)
        child.brush = child_brush

        return child



class Canvas:
    def __init__(self, img, num_sections):
        self.height, self.width = img.shape[:2]
        self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        self.num_sections = num_sections  # Se guarda el número de secciones
        print("Tamaño del lienzo:", self.width, "x", self.height)

    def paint(self, population):
        for individual in population:
            paste_x, paste_y = individual.xy_position

            # Aplica la brocha usando la máscara para evitar el fondo
            brush_height, brush_width = individual.brush.shape[:2]
            mask_inv = cv2.bitwise_not(individual.mask)
            roi = self.canvas[paste_y:paste_y + brush_height, paste_x:paste_x + brush_width]
            individual.brush = cv2.bitwise_and(individual.brush, individual.brush, mask=individual.mask)
            bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            dst = cv2.add(bg, individual.brush)

            # Actualiza el lienzo con la brocha aplicada
            self.canvas[paste_y:paste_y + brush_height, paste_x:paste_x + brush_width] = dst

        # Muestra el lienzo después de pintar toda la población
        cv2.imshow('Image', self.canvas)
        cv2.waitKey(0)


class GeneticAlgorithm:
    def __init__(self, target_image, population_size, max_generations, num_sections):
        self.target_image = target_image
        self.population_size = population_size
        self.max_generations = max_generations
        self.num_sections = num_sections
        self.brushes_list = load_brushes()
        self.current_population = []
        self.canvas = Canvas(target_image, num_sections)

    def initialize_population(self):
        for _ in range(self.population_size):
            new_individual = DNA(self.target_image, self.brushes_list, self.canvas)
            new_individual.generate_random_position(self.canvas.height, self.canvas.width)
            self.current_population.append(new_individual)
        self.canvas.paint(self.current_population)

    def evolve(self):
        for generation in range(self.max_generations):
            for individual in self.current_population:
                individual.calculate_fitness()
                print(f"Fitness: {individual.fitness}")

            elite = self.selection()
                # Inicializar la nueva generación con la élite
            new_generation = elite.copy()

            # Aplicar la reproducción (cruce) y mutación a la población
            while len(new_generation) < self.population_size:
                # Selección de padres
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)

                # Cruce
                child = parent1.crossover(parent1, parent2, self.brushes_list)


                # Mutación
                child.mutate(self.canvas)

                # Agregar hijo a la nueva generación
                new_generation.append(child)

            # Actualizar la población actual con la nueva generación
            self.current_population = new_generation

            # Pintar la población en el lienzo
            self.canvas.paint(self.current_population)

    def selection(self, rate=0.2):
        # Ordenar la población por aptitud de mayor a menor
        sorted_population = sorted(self.current_population, key=lambda x: x.fitness, reverse=True)
        # Seleccionar el porcentaje especificado de individuos con mejor aptitud
        elite_count = int(self.population_size * rate)
        elite = sorted_population[:elite_count]
        return elite


def start(target_image_path, population_size, max_generations, num_sections):
    img = cv2.imread(target_image_path)

    genetic_algorithm = GeneticAlgorithm(img, population_size, max_generations, num_sections)
    genetic_algorithm.initialize_population()
    genetic_algorithm.evolve()


if __name__ == "__main__":
    target_image_path = "Images/PALETA.jpg"
    population_size = 15
    max_generations = 10
    num_sections = 20  # Número de secciones en cada dimensión
    start(target_image_path, population_size, max_generations, num_sections)
