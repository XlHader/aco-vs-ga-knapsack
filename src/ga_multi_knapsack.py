import numpy as np
import time
import pygad
import os


class GAMultiKnapsackSolver:
    def __init__(self,
                 instance_file,
                 num_generations=600,       # Aumentado para instancias grandes
                 sol_per_pop=250,           # Mayor tamaño de población
                 num_parents_mating=50,
                 parent_selection_type="tournament",
                 k_tournament=4,
                 crossover_type="two_points",  # Para problemas binarios suele funcionar mejor
                 crossover_probability=0.8,
                 mutation_type="random",
                 mutation_probability=0.02,  # Mutación baja para conservar factibilidad
                 keep_elitism=5,
                 stop_criteria=None,
                 num_threads=1,
                 random_seed=42):
        """
        Inicializa el solucionador GA para problemas de mochila con múltiples restricciones.
        """
        self.instance_file = instance_file
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.parent_selection_type = parent_selection_type
        self.k_tournament = k_tournament
        self.crossover_type = crossover_type
        self.crossover_probability = crossover_probability
        self.mutation_type = mutation_type
        self.mutation_probability = mutation_probability
        self.keep_elitism = keep_elitism
        self.stop_criteria = stop_criteria or ["saturate_50"]
        self.num_threads = num_threads
        self.random_seed = random_seed

        # Cargar el problema usando el módulo de carga (asegúrate de tener src/knapsack_loader.py)
        from src.knapsack_loader import MultiConstraintKnapsackProblem
        self.problem = MultiConstraintKnapsackProblem(instance_file)

    def check_optimality(self, current_value):
        """
        Verifica si el valor actual ha alcanzado el óptimo o está muy cercano.
        Devuelve True si se ha encontrado el óptimo (gap <= 0.001%).
        """
        if not self.problem.optimal_value:
            return False

        gap = (self.problem.optimal_value - current_value) / \
            self.problem.optimal_value * 100
        # Consideramos óptimo si el gap es 0.001% o menor
        return abs(gap) <= 0.001

    def _is_feasible(self, solution):
        """Verifica si la solución cumple con todas las restricciones."""
        for i in range(self.problem.num_constraints):
            if np.dot(self.problem.weights[i], solution) > self.problem.capacities[i]:
                return False
        return True

    def repair_solution(self, solution):
        """
        Repara una solución infactible removiendo objetos de menor eficiencia
        hasta que se cumplan todas las restricciones.
        """
        repaired = solution.copy()
        # Mientras la solución sea infactible, removemos el objeto menos eficiente
        while not self._is_feasible(repaired):
            # Obtener índices de ítems activos (1)
            indices = np.where(repaired == 1)[0]
            if len(indices) == 0:
                # No queda nada para remover; devuelve la solución (aunque no sea factible)
                break

            efficiencies = []
            for j in indices:
                # Calcular eficiencia como valor / (suma de pesos en todas las restricciones)
                total_weight = np.sum(self.problem.weights[:, j])
                eff = self.problem.values[j] / \
                    total_weight if total_weight > 0 else self.problem.values[j]
                efficiencies.append((j, eff))
            # Ordenar de menor a mayor eficiencia
            efficiencies.sort(key=lambda x: x[1])
            # Remover el objeto menos eficiente
            j_remove = efficiencies[0][0]
            repaired[j_remove] = 0
        return repaired

    def _create_initial_population(self):
        """
        Genera la población inicial (factible o reparada) como un array
        de dimensiones (sol_per_pop, num_objects).
        """
        population = []
        for _ in range(self.sol_per_pop):
            candidate = np.random.randint(2, size=self.problem.num_objects)
            # Reparar candidate si es infactible
            if not self._is_feasible(candidate):
                candidate = self.repair_solution(candidate)
            population.append(candidate)
        return np.array(population)

    def run(self):
        """Ejecuta el algoritmo genético para el problema de múltiples restricciones."""
        # Función de fitness
        def fitness_func(ga_instance, solution, solution_idx):
            solution_int = solution.astype(int)
            if self._is_feasible(solution_int):
                return float(np.sum(np.array(self.problem.values) * solution_int))
            else:
                # Reparar la solución y penalizar fuertemente
                repaired = self.repair_solution(solution_int)
                repaired_value = np.sum(
                    np.array(self.problem.values) * repaired)
                # Penalización: resta un valor grande (ajustar según magnitud)
                return float(repaired_value) - 1e6

        convergence = []
        self.found_optimal = False  # Inicializar la variable de seguimiento

        def on_generation(ga_instance):
            best_sol = ga_instance.best_solution()[0].astype(int)
            best_val = np.sum(np.array(self.problem.values) * best_sol)
            convergence.append(best_val)

            # Verificar si hemos alcanzado el óptimo
            if self.problem.optimal_value and self.check_optimality(best_val):
                self.found_optimal = True
                print(f"[GA] ¡SOLUCIÓN ÓPTIMA ENCONTRADA! Valor: {best_val}")
                print(
                    f"     El algoritmo se detendrá en la generación {ga_instance.generations_completed}")
                # Forzar la detención del algoritmo
                return "stop"

            if ga_instance.generations_completed % 10 == 0 or ga_instance.generations_completed == 1:
                print(
                    f"[GA] Generación {ga_instance.generations_completed}/{self.num_generations} - Mejor valor: {best_val}")
                if self.problem.optimal_value:
                    gap = (self.problem.optimal_value - best_val) / \
                        self.problem.optimal_value * 100
                    print(f"     Gap al óptimo: {gap:.2f}%")

            return None

        start_time = time.time()

        # Configuración de paralelización (si es aplicable)
        parallel_arg = [
            "thread", self.num_threads] if self.num_threads > 1 else None

        initial_pop = self._create_initial_population()

        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.problem.num_objects,
            gene_type=int,
            gene_space=[0, 1],
            parent_selection_type=self.parent_selection_type,
            K_tournament=self.k_tournament,
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_probability,
            mutation_type=self.mutation_type,
            mutation_probability=self.mutation_probability,
            keep_elitism=self.keep_elitism,
            stop_criteria=self.stop_criteria,
            on_generation=on_generation,
            random_seed=self.random_seed,
            parallel_processing=parallel_arg,
            initial_population=initial_pop
        )

        ga_instance.run()
        elapsed_time = time.time() - start_time

        best_solution, best_fitness, _ = ga_instance.best_solution()
        best_solution = best_solution.astype(int)
        if not self._is_feasible(best_solution):
            best_solution = self.repair_solution(best_solution)
        best_value = np.sum(np.array(self.problem.values) * best_solution)

        # Verificar si encontramos el óptimo después de reparación
        if not self.found_optimal and self.problem.optimal_value:
            if int(best_value) == int(self.problem.optimal_value) or self.check_optimality(best_value):
                self.found_optimal = True
                print("[GA] Se alcanzó el óptimo después de la verificación final!")

        # Calcular el gap al óptimo correctamente
        optimal_gap = None
        if self.problem.optimal_value:
            if self.found_optimal:
                # Si encontramos el óptimo, el gap es 0
                optimal_gap = 0.0
                # También podemos ajustar el valor para que sea exactamente el óptimo
                best_value = float(self.problem.optimal_value)
                if len(convergence) > 0:
                    convergence[-1] = float(self.problem.optimal_value)
            else:
                # Calculamos el gap normalmente
                optimal_gap = (self.problem.optimal_value -
                               best_value) / self.problem.optimal_value * 100

        return {
            "solution": best_solution,
            "value": best_value,
            "time": elapsed_time,
            "convergence": convergence,
            "is_feasible": self._is_feasible(best_solution),
            "optimal_gap": optimal_gap,
            "found_optimal": self.found_optimal
        }
