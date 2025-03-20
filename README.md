# Créditos Data  
Las instancias de problemas de mochila utilizadas fueron contribuidas a OR-Library por Joerg Heitkoetter.  
Más información: [OR-Library Multiple Knapsack Problems](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap2.txt)

# ACO vs GA - Comparación de Algoritmos para el Problema de la Mochila

Este proyecto implementa y compara dos algoritmos de optimización para resolver problemas de mochila con múltiples restricciones utilizando instancias en formato WEING (y otros formatos clásicos). Los algoritmos implementados son:

- **ACO (Optimización de Colonias de Hormigas)**
- **GA (Algoritmos Genéticos)**

Además, se ha optimizado el rendimiento mediante paralelización configurable.

---

## Características

- **Implementación modular:** Clases separadas para cada algoritmo (ACO y GA) y un módulo dedicado para cargar instancias (knapsack_loader).
- **Configuración flexible:** Parámetros ajustables mediante argumentos de línea de comandos.
- **Ejecución en paralelo:** Configurable con el parámetro `--threads`.
- **Gestión de soluciones infactibles:** Penalización agresiva y reparación iterativa en el GA para forzar soluciones factibles.
- **Comparativa de convergencia:** Se generan gráficos de convergencia y una tabla resumen con valor obtenido, tiempo de ejecución y gap al óptimo.

---

## Requisitos

- **Python 3.10+**
- **Dependencias:**  
  Instala las dependencias con:
  ```sh
  pip install -r requirements.txt
  ```
- **Instancias de Problema:**  
  Coloca tus archivos de instancia (por ejemplo, WEING1.DAT, WEING2.DAT, etc.) en la carpeta `knapsack_data/`.

---

## Uso

### Ejecución Manual

Ejemplo de ejecución para una instancia (por ejemplo, WEING6.DAT):

#### Buena configuración del 1 al 6

```sh
python3 -m src.main --instance knapsack_data/WEING1.DAT \
  --threads 4 \
  --aco_ants 100 --aco_iterations 200 --aco_stagnation 60 \
  --ga_generations 200 --ga_pop_size 220 --ga_stop_criteria saturate_150 --ga_crossover two_points --ga_mutation 0.05 --ga_k_tournament 4
```

#### Buena configuración 7

```sh
python3 -m src.main --instance knapsack_data/WEING7.DAT \
  --threads 4 \
  --aco_ants 100 --aco_iterations 500 --aco_stagnation 150 \
  --ga_generations 700 --ga_pop_size 260 --ga_stop_criteria saturate_250 --ga_crossover two_points --ga_mutation 0.01 --ga_k_tournament 2
```

#### Buena configuración 8

```sh
python3 -m src.main --instance knapsack_data/WEING8.DAT \
  --threads 4 \
  --aco_ants 120 --aco_iterations 800 --aco_stagnation 200 \
  --ga_generations 800 --ga_pop_size 300 --ga_stop_criteria saturate_300 --ga_crossover two_points --ga_mutation 0.01 --ga_k_tournament 4
```

### Parámetros Clave

- `--instance`: Ruta al archivo de instancia (formato WEING u otro).
- `--threads`: Número de hilos para la paralelización (usado en ACO y GA).
- **Parámetros ACO:**
  - `--aco_ants`: Número de hormigas.
  - `--aco_iterations`: Número de iteraciones.
  - `--aco_alpha`, `--aco_beta`: Influencia de feromonas y heurística.
  - `--aco_evaporation`: Tasa de evaporación.
  - `--aco_stagnation`: Iteraciones máximas sin mejora.
- **Parámetros GA:**
  - `--ga_generations`: Número de generaciones.
  - `--ga_pop_size`: Tamaño de población.
  - `--ga_mutation_type`: Tipo de mutación (e.g., random, swap, inversion).
  - `--ga_mutation`: Probabilidad de mutación.
  - `--ga_crossover`: Probabilidad de cruce.
  - `--ga_stop_criteria`: Criterio de parada (e.g., saturate_50).
  - `--ga_k_tournament`: Número de individuos en torneo para selección.

---

## Estructura del Proyecto

```
aco
 ┣ knapsack_data
 ┃ ┣ WEING1.DAT
 ┃ ┣ WEING2.DAT
 ┃ ┣ WEING3.DAT
 ┃ ┣ WEING4.DAT
 ┃ ┣ WEING5.DAT
 ┃ ┣ WEING6.DAT
 ┃ ┣ WEING7.DAT
 ┃ ┗ WEING8.DAT
 ┣ src
 ┃ ┣ __init__.py
 ┃ ┣ aco_multi_knapsack.py
 ┃ ┣ ga_multi_knapsack.py
 ┃ ┣ knapsack_loader.py
 ┃ ┗ main.py
 ┣ README.md
 ┗ requirements.txt
```

---

## Notas Adicionales

- Los resultados (gráficos y CSV) se guardan en subcarpetas dentro de `knapsack_solutions/solution_<instance_name>/<timestamp>`.
- Asegúrate de tener permisos de escritura en el directorio del proyecto.
- La ejecución en paralelo se controla mediante el parámetro `--threads`.