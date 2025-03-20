import numpy as np
import time
import concurrent.futures


class ACOMultiKnapsackSolver:
    def __init__(self,
                 instance_file,
                 num_ants=10,
                 num_iterations=100,
                 alpha=1.0,
                 beta=2.0,
                 evaporation_rate=0.1,
                 q=1.0,
                 num_threads=1,
                 max_stagnation=20):
        """
        Inicializa el solucionador ACO para problemas de mochila con múltiples restricciones.
        """
        self.instance_file = instance_file
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.num_threads = num_threads
        self.max_stagnation = max_stagnation

        # Cargar el problema
        from src.knapsack_loader import MultiConstraintKnapsackProblem
        self.problem = MultiConstraintKnapsackProblem(instance_file)

        # Inicializar feromonas - una por objeto
        self.pheromones = np.ones(self.problem.num_objects)

        # Calcular heurística (valor/suma de pesos)
        self.heuristic = np.zeros(self.problem.num_objects)
        for j in range(self.problem.num_objects):
            # Para cada objeto, sumar todos sus pesos en todas las restricciones
            total_weight = np.sum(self.problem.weights[:, j])
            if total_weight > 0:
                self.heuristic[j] = self.problem.values[j] / total_weight
            else:
                # Si el peso es 0, usamos el valor
                self.heuristic[j] = self.problem.values[j]

    def _is_feasible(self, solution):
        """Verifica si una solución es factible (cumple con todas las restricciones)"""
        for i in range(self.problem.num_constraints):
            if np.dot(self.problem.weights[i], solution) > self.problem.capacities[i]:
                return False
        return True

    def construct_solution(self, ant_id):
        """Una hormiga construye una solución para el problema de múltiples restricciones"""
        # Inicializar con todos los objetos no seleccionados
        solution = np.zeros(self.problem.num_objects, dtype=int)

        # Lista de objetos disponibles para seleccionar
        available_objects = np.ones(self.problem.num_objects, dtype=bool)

        # Recursos disponibles (capacidad restante para cada restricción)
        remaining_capacity = np.array(self.problem.capacities.copy())

        # Mientras haya objetos disponibles
        while np.any(available_objects):
            # Calcular probabilidades para los objetos disponibles
            probabilities = np.zeros(self.problem.num_objects)

            for j in range(self.problem.num_objects):
                if available_objects[j]:
                    # Verificar si agregar este objeto excede alguna capacidad
                    would_exceed = False
                    for i in range(self.problem.num_constraints):
                        if self.problem.weights[i, j] > remaining_capacity[i]:
                            would_exceed = True
                            break

                    if not would_exceed:
                        # Calcular la probabilidad según fórmula ACO
                        probabilities[j] = (
                            self.pheromones[j] ** self.alpha) * (self.heuristic[j] ** self.beta)

            # Si no hay objetos válidos para seleccionar, terminamos
            if np.sum(probabilities) == 0:
                break

            # Normalizar probabilidades
            probabilities = probabilities / np.sum(probabilities)

            # Seleccionar un objeto usando la ruleta
            cumulative_prob = np.cumsum(probabilities)
            r = np.random.random()
            for j in range(self.problem.num_objects):
                if r <= cumulative_prob[j]:
                    selected_object = j
                    break

            # Actualizar la solución y capacidades restantes
            solution[selected_object] = 1
            for i in range(self.problem.num_constraints):
                remaining_capacity[i] -= self.problem.weights[i,
                                                              selected_object]

            # Marcar el objeto como no disponible
            available_objects[selected_object] = False

        # Calcular el valor total de la solución
        solution_value = np.sum(np.array(self.problem.values) * solution)

        return solution, solution_value

    def local_search(self, solution):
        """Mejora local de la solución: intenta añadir objetos no seleccionados si es posible"""
        improved_solution = solution.copy()
        solution_value = np.sum(
            np.array(self.problem.values) * improved_solution)

        # Intentar agregar objetos no seleccionados
        for j in range(self.problem.num_objects):
            if improved_solution[j] == 0:
                # Verificar si podemos agregar este objeto
                improved_solution[j] = 1

                if self._is_feasible(improved_solution):
                    # Si es factible, lo mantenemos
                    solution_value += self.problem.values[j]
                else:
                    # Si no es factible, lo quitamos
                    improved_solution[j] = 0

        return improved_solution, solution_value

    def run(self):
        """Ejecuta el algoritmo ACO para el problema de múltiples restricciones"""
        best_solution = None
        best_value = 0
        stagnation_count = 0
        convergence = []

        start_time = time.time()

        for iteration in range(self.num_iterations):
            # Construir soluciones en paralelo si se especifica
            if self.num_threads > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    ant_solutions = list(executor.map(
                        self.construct_solution, range(self.num_ants)))
            else:
                ant_solutions = [self.construct_solution(
                    ant) for ant in range(self.num_ants)]

            # Aplicar búsqueda local a cada solución
            improved_solutions = []
            for solution, value in ant_solutions:
                improved_solution, improved_value = self.local_search(solution)
                improved_solutions.append((improved_solution, improved_value))

            # Encontrar la mejor solución de la iteración
            iteration_best_solution = None
            iteration_best_value = 0

            for solution, value in improved_solutions:
                if value > iteration_best_value and self._is_feasible(solution):
                    iteration_best_value = value
                    iteration_best_solution = solution

            # Actualizar la mejor solución global
            if iteration_best_value > best_value:
                best_value = iteration_best_value
                best_solution = iteration_best_solution.copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Evaporar feromonas
            self.pheromones = (1 - self.evaporation_rate) * self.pheromones

            # Actualizar feromonas con la mejor solución de la iteración
            if iteration_best_solution is not None:
                delta_tau = self.q / (1 + iteration_best_value)
                for j in range(self.problem.num_objects):
                    if iteration_best_solution[j] == 1:
                        self.pheromones[j] += delta_tau

            # Registrar convergencia
            convergence.append(best_value)

            # Imprimir progreso
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(
                    f"[ACO] Iteración {iteration+1}/{self.num_iterations} - Mejor valor: {best_value}")

                # Mostrar distancia al óptimo si está disponible
                if self.problem.optimal_value:
                    gap = (self.problem.optimal_value - best_value) / \
                        self.problem.optimal_value * 100
                    print(f"      Gap al óptimo: {gap:.2f}%")

            # Verificar criterio de parada por estancamiento
            if self.max_stagnation and stagnation_count >= self.max_stagnation:
                print(
                    f"[ACO] Detenido por estancamiento después de {stagnation_count} iteraciones")
                break

        elapsed_time = time.time() - start_time

        # Verificar restricciones de la mejor solución
        is_feasible = self._is_feasible(best_solution)

        if not is_feasible:
            print("[ACO] ADVERTENCIA: La mejor solución encontrada no es factible.")

        return {
            "solution": best_solution,
            "value": best_value,
            "time": elapsed_time,
            "convergence": convergence,
            "is_feasible": is_feasible,
            "optimal_gap": (self.problem.optimal_value - best_value) / self.problem.optimal_value * 100 if self.problem.optimal_value else None
        }
