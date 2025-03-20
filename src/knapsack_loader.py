import numpy as np


class MultiConstraintKnapsackProblem:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load()

    def load(self):
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        # Eliminar comentarios y líneas vacías
        lines = [line.strip() for line in lines if line.strip()
                 and not line.strip().startswith('//')]

        # Primera línea: número de restricciones (knapsacks) y número de objetos
        first_line = lines[0].split()
        self.num_constraints = int(first_line[0])
        self.num_objects = int(first_line[1])

        # Líneas siguientes: valores (profits) de los objetos
        values_line_count = (self.num_objects + 9) // 10  # Ceiling division
        self.values = []
        for i in range(1, 1 + values_line_count):
            self.values.extend(int(x) for x in lines[i].split())

        # Líneas siguientes: capacidades de las mochilas
        capacities_line_count = (
            self.num_constraints + 9) // 10  # Ceiling division
        line_idx = 1 + values_line_count
        self.capacities = []
        for i in range(line_idx, line_idx + capacities_line_count):
            self.capacities.extend(int(x) for x in lines[i].split())

        # Líneas siguientes: matriz de restricciones
        line_idx = line_idx + capacities_line_count
        self.weights = np.zeros((self.num_constraints, self.num_objects))

        for i in range(self.num_constraints):
            constraint_values = []
            constraint_line_count = (
                self.num_objects + 9) // 10  # Ceiling division
            for j in range(line_idx, line_idx + constraint_line_count):
                if j < len(lines):
                    constraint_values.extend(int(x) for x in lines[j].split())
            line_idx += constraint_line_count
            self.weights[i, :len(constraint_values)] = constraint_values

        # Última línea: valor óptimo conocido
        if line_idx < len(lines):
            self.optimal_value = int(lines[line_idx])
        else:
            self.optimal_value = None
