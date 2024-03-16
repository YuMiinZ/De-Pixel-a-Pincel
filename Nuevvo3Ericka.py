from PIL import Image
import numpy as np
import os
import random
import cv2
import math

def load_brushes():
    brushes = []
    for filename in os.listdir('Brushes/'):
        brush = cv2.imread('Brushes/' + filename, cv2.IMREAD_GRAYSCALE)
        brushes.append(brush)
    return brushes


def extract_color(matrix):
    # Implementa la lógica para extraer el color dominante de una matriz de imagen
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
            end_row = (i + 1) * section_height
            start_col = j * section_width
            end_col = (j + 1) * section_width

            section = image[start_row:end_row, start_col:end_col]
            print("Tamaño de la sección en", i, ",", j, ":", section.shape[0], "x", section.shape[1])  # Aquí agregamos el print
            sections.append(section)

    return sections


class DNA:
    def __init__(self, target_image, brushes_list, canvas, sections):
        self.target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        self.brush = self.resize_brush(random.choice(brushes_list), 0.1, 0.3)
        self.brush, self.mask = self.change_brush_color()
        self.fitness = None
        self.xy_position = None
        self.canvas = canvas  # Se guarda una referencia al lienzo
        self.sections = sections  # Se guarda una referencia a las secciones de la imagen
        print("Tamaño de la brocha:", self.brush.shape)

    def resize_brush(self, brush, minRange, maxRange):
        new_height = random.randint(int(brush.shape[0] * minRange), int(brush.shape[0] * maxRange))
        new_width = random.randint(int(brush.shape[1] * minRange), int(brush.shape[1] * maxRange))

        new_height = max(1, new_height)
        new_width = max(1, new_width)
        return cv2.resize(brush, (new_width, new_height))

    def change_brush_color(self):
        # Generar un tono de gris aleatorio
        random_gray = random.randint(255, 255)
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
        print("Número de secciones:", num_sections)
        for i in range(num_sections):
            for j in range(num_sections):
                print("Índices i, j:", i, j)
                section_x = j * (self.canvas.width // num_sections)
                section_y = i * (self.canvas.height // num_sections)
                
                # Verificar que los índices estén dentro de los límites de la matriz canvas.canvas
                if section_y + self.sections[i][j].shape[0] <= self.canvas.height and \
                        section_x + self.sections[i][j].shape[1] <= self.canvas.width:
                    
                    canvas_section = self.canvas.canvas[section_y:section_y + self.sections[i][j].shape[0],
                                                        section_x:section_x + self.sections[i][j].shape[1]]

                    # Obtener el tamaño de la sección de la imagen y calcular la similitud
                    section_height, section_width = self.sections[i][j].shape[:2]
                    print("Tamaño de la sección:", section_height, "x", section_width)

                    similarity = calculate_similarity(canvas_section, self.sections[i][j])
                    total_similarity += similarity
                else:
                    print("Sección fuera de los límites de la matriz canvas.canvas")

        self.fitness = total_similarity / num_sections ** 2


    def mutate(self, canvas):
        # Implementa la lógica para mutar un individuo
        pass

    def crossover(self, parent1, parent2):
        child = DNA(self.target_image, self.brushes_list, self.canvas, self.sections)
        crossover_point = random.randint(0, min(len(parent1.brush), len(parent2.brush)))
        child.brush = np.concatenate((parent1.brush[:crossover_point], parent2.brush[crossover_point:]))
        return child


class Canvas:
    def __init__(self, img, num_sections, width, height):
        self.height = height
        self.width = width
        print("Tamaño de la imagen original:", width, "x", height)  # Aquí agregamos el print
        self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        self.num_sections = num_sections  # Se guarda el número de secciones
        self.sections = divide_image_into_sections(img, num_sections)  # Divide la imagen en secciones
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
            self.canvas[paste_y:paste_y + brush_height, paste_x:paste_x + brush_width] = dst

            # individual.calculate_fitness()  # Calcular la similitud con la imagen original

            cv2.imshow('Image', self.canvas)
        cv2.waitKey(0)


class GeneticAlgorithm:
    def __init__(self, target_image, population_size, max_generations, num_sections, image_width, image_height):
        self.target_image = target_image
        self.population_size = population_size
        self.max_generations = max_generations
        self.num_sections = num_sections
        self.brushes_list = load_brushes()
        self.current_population = []
        self.canvas = Canvas(target_image, num_sections, image_width, image_height)

    def initialize_population(self):
        for _ in range(self.population_size):
            new_individual = DNA(self.target_image, self.brushes_list, self.canvas, self.canvas.sections)
            new_individual.generate_random_position(self.canvas.height, self.canvas.width)
            self.current_population.append(new_individual)
        self.canvas.paint(self.current_population)
        self.evolve()

    def evolve(self):
        print(f"Comenzando evolución")
        for _ in range(self.max_generations):
            for individual in self.current_population:
                individual.calculate_fitness()
                print(f"Fitness del individuo: {individual.fitness}")

            # Aquí podrías agregar la lógica para la selección, cruce y mutación de la población
            #groups_to_evolve = self.select_groups_to_evolve()
            # self.evolve_selected_groups(groups_to_evolve, num_children=1)

            # Pintar la población en el lienzo
            self.canvas.paint(self.current_population)

    def select_groups_to_evolve(self):
        groups_to_evolve = []
        for group_index in range(self.num_sections):
            group_fitness = self.calculate_group_fitness(group_index)
            if group_fitness < 0.8:  # Si la similitud del grupo es menor al 80%
                groups_to_evolve.append(group_index)
        return groups_to_evolve

    def calculate_group_fitness(self, group_index):
        total_similarity = 0
        num_sections = self.canvas.num_sections
        for i in range(num_sections):
            for j in range(num_sections):
                section_x, section_y = self.get_section_coordinates(group_index)
                canvas_section = self.canvas.canvas[section_y:section_y + self.canvas.sections[i][j].shape[0],
                                                    section_x:section_x + self.canvas.sections[i][j].shape[1]]
                similarity = calculate_similarity(canvas_section, self.canvas.sections[i][j])
                total_similarity += similarity
        return total_similarity / num_sections ** 2

    def evolve_selected_groups(self, groups_to_evolve, num_children):
        for group_index in groups_to_evolve:
            # Seleccionar individuos aleatorios de este grupo para evolucionar
            group_individuals = [individual for individual in self.current_population
                                 if individual.xy_position in self.get_section_coordinates(group_index)]
            selected_individuals = random.sample(group_individuals, num_children)

            # Generar nuevos individuos (hijos) mediante cruce y mutación
            children = []
            for _ in range(num_children):
                parent1, parent2 = random.sample(selected_individuals, 2)
                child.crossover(parent1, parent2)
                child.mutate(self.canvas)
                children.append(child)

            # Asignar nuevas posiciones a los hijos
            for child in children:
                child.generate_random_position(self.canvas.height, self.canvas.width)
                self.current_population.append(child)

    def get_section_coordinates(self, group_index):
        sections_per_row = int(math.sqrt(self.num_sections))
        section_height = self.canvas.height // sections_per_row
        section_width = self.canvas.width // sections_per_row

        row = group_index // sections_per_row
        col = group_index % sections_per_row

        start_x = col * section_width
        start_y = row * section_height

        return start_x, start_y


def load_image_size(image_path):
    # Abre la imagen con PIL y obtiene su tamaño
    image = Image.open(image_path)
    width, height = image.size
    return width, height


def start(target_image_path, population_size, max_generations, num_sections):
    image_width, image_height = load_image_size(target_image_path)

    img = cv2.imread(target_image_path)
    genetic_algorithm = GeneticAlgorithm(img, population_size, max_generations, num_sections, image_width, image_height)
    genetic_algorithm.initialize_population()


if __name__ == "__main__":
    target_image_path = "Images/PALETA.jpg"
    population_size = 1
    max_generations = 1
    num_sections = 20  # Número de secciones en cada dimensión
    start(target_image_path, population_size, max_generations, num_sections)
