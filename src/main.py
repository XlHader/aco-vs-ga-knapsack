import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparar ACO y GA para problemas de mochila con múltiples restricciones")

    parser.add_argument("--instance", type=str, required=True,
                        help="Ruta al archivo de instancia (formato WEING)")

    # Parámetros generales
    parser.add_argument("--threads", type=int, default=4,
                        help="Número de hilos para paralelización")

    # Parámetros ACO
    parser.add_argument("--aco_ants", type=int, default=20,
                        help="Número de hormigas")
    parser.add_argument("--aco_iterations", type=int, default=100,
                        help="Número de iteraciones ACO")
    parser.add_argument("--aco_alpha", type=float, default=1.0,
                        help="Parámetro alpha (influencia de feromonas)")
    parser.add_argument("--aco_beta", type=float, default=2.0,
                        help="Parámetro beta (influencia de heurística)")
    parser.add_argument("--aco_evaporation", type=float, default=0.1,
                        help="Tasa de evaporación de feromonas")
    parser.add_argument("--aco_stagnation", type=int, default=30,
                        help="Iteraciones máximas sin mejora")

    # Parámetros GA
    parser.add_argument("--ga_generations", type=int, default=100,
                        help="Número de generaciones GA")
    parser.add_argument("--ga_pop_size", type=int, default=50,
                        help="Tamaño de población GA")
    parser.add_argument("--ga_mutation_type", type=str, default="random",
                        choices=["random", "swap", "inversion",
                                 "scramble", "adaptive"],
                        help="Tipo de mutación GA")
    parser.add_argument("--ga_mutation", type=float, default=0.1,
                        help="Probabilidad de mutación GA")
    parser.add_argument("--ga_crossover", type=str, default="scattered",
                        help="Tipo de crossover en GA ('scattered', 'two_points')")
    parser.add_argument("--ga_crossover_prob", type=float, default=0.8,
                        help="Probabilidad de cruce GA")
    parser.add_argument("--ga_stop_criteria", type=str, default="saturate_150",
                        help="Criterio de parada en GA (Ejemplo: 'saturate_150)")
    parser.add_argument("--ga_k_tournament", type=int, default=3,
                        help="Número de individuos en el torneo para selección")

    return parser.parse_args()


def main():
    args = parse_args()

    instance_name = os.path.basename(args.instance)
    print(f"\n=== Resolviendo instancia: {instance_name} ===\n")

    # Ejecutar ACO
    print("Ejecutando ACO...")
    from src.aco_multi_knapsack import ACOMultiKnapsackSolver

    aco_solver = ACOMultiKnapsackSolver(
        instance_file=args.instance,
        num_ants=args.aco_ants,
        num_iterations=args.aco_iterations,
        alpha=args.aco_alpha,
        beta=args.aco_beta,
        evaporation_rate=args.aco_evaporation,
        num_threads=args.threads,
        max_stagnation=args.aco_stagnation
    )

    aco_results = aco_solver.run()

    # Ejecutar GA
    print("\nEjecutando GA...")
    from src.ga_multi_knapsack import GAMultiKnapsackSolver

    ga_solver = GAMultiKnapsackSolver(
        instance_file=args.instance,
        num_generations=args.ga_generations,
        sol_per_pop=args.ga_pop_size,
        mutation_type=args.ga_mutation_type,
        mutation_probability=args.ga_mutation,
        crossover_type=args.ga_crossover,
        crossover_probability=args.ga_crossover_prob,
        stop_criteria=args.ga_stop_criteria,
        k_tournament=args.ga_k_tournament,
        num_threads=args.threads,
    )

    ga_results = ga_solver.run()

    # Crear tabla comparativa
    optimal_value = aco_solver.problem.optimal_value or "Desconocido"

    results_df = pd.DataFrame({
        'Algoritmo': ['ACO', 'GA'],
        'Valor': [aco_results['value'], ga_results['value']],
        'Tiempo (s)': [round(aco_results['time'], 2), round(ga_results['time'], 2)],
        'Factible': [aco_results['is_feasible'], ga_results['is_feasible']],
        'Gap al óptimo (%)': [
            round(aco_results['optimal_gap'],
                  2) if aco_results['optimal_gap'] is not None else 'N/A',
            round(ga_results['optimal_gap'],
                  2) if ga_results['optimal_gap'] is not None else 'N/A'
        ]
    })

    print("\n=== Resultados ===")
    print(f"Problema: {instance_name}")
    print(f"Valor óptimo: {optimal_value}")
    print("\nComparación de algoritmos:")
    print(results_df)

    # Graficar convergencia
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(aco_results['convergence'])
    if aco_solver.problem.optimal_value:
        plt.axhline(y=aco_solver.problem.optimal_value,
                    color='r', linestyle='--', label='Óptimo')
    plt.title(f"Convergencia ACO - {instance_name}")
    plt.xlabel("Iteración")
    plt.ylabel("Valor")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(ga_results['convergence'])
    if ga_solver.problem.optimal_value:
        plt.axhline(y=ga_solver.problem.optimal_value,
                    color='r', linestyle='--', label='Óptimo')
    plt.title(f"Convergencia GA - {instance_name}")
    plt.xlabel("Generación")
    plt.ylabel("Valor")
    plt.legend()

    plt.tight_layout()

    # Guardar gráfico y resultados
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = f"knapsack_solutions/solution_{instance_name}/{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    plt.savefig(os.path.join(output_folder, f"{timestamp}.png"))
    results_df.to_csv(os.path.join(
        output_folder, f"{timestamp}.csv"), index=False)

    print(f"\nGráficos y resultados guardados con timestamp {timestamp}")
    plt.show()


if __name__ == "__main__":
    main()
