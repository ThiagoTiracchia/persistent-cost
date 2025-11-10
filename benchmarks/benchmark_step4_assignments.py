import csv
import os
import time
import tracemalloc
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform

try:
    from persistent_cost.cylinder import step1
    from benchmark_cylinder import random_sparse_matrix
    from persistent_cost.algorithms.sparse_fast import do_pivot_cython as do_pivot
except Exception as e:

    step1 = None
    do_pivot = None


def benchmark_assignments(D_f, V_g, R_g, mapping_L, formato_test='lil'):
    """
    Benchmarkea específicamente las asignaciones elemento por elemento de step4.
    
    Basado en step4 de cylinder.py, mide el tiempo y memoria de las asignaciones,
    y luego cuenta qué tipo de asignaciones se hicieron.
    
    Args:
        D_f: Matriz dispersa del complejo K
        V_g: Matriz de transformación de D_g
        R_g: Matriz reducida de D_g
        mapping_L: Lista de índices de COLUMNAS en D_f correspondientes a L
        formato_test: Formato de matriz a usar para D_cok ('lil', 'dok', 'csr', 'csc')
    
    Returns:
        dict con métricas del benchmark
    """
    
    # Métricas iniciales
    nnz_before = D_f.getnnz()
    
    # Copia en formato especificado para las asignaciones
    if formato_test == 'lil':
        D_cok = D_f.tolil()
    elif formato_test == 'dok':
        D_cok = D_f.todok()
    elif formato_test == 'csr':
        D_cok = D_f.tocsr() # Conversión intermedia para asignaciones
    elif formato_test == 'csc':
        D_cok = D_f.tocsc()  # Conversión intermedia para asignaciones
    else:
        D_cok = D_f.tolil()

    # Snapshot de coordenadas existentes ANTES de las asignaciones
    existing_coords_before = set()
    D_f_coo = D_f.tocoo()
    for i, j in zip(D_f_coo.row, D_f_coo.col):
        existing_coords_before.add((int(i), int(j)))

    # Columnas de R_g que son ciclos (columnas con todo cero)
    cycle_columns = [c for c in range(R_g.shape[1]) if R_g[:, c].getnnz() == 0]

    # Índices en D_f correspondientes a esas columnas de ciclos
    index_cycle_columns_f = [mapping_L[c] for c in cycle_columns if c < len(mapping_L)]

    # Usamos todas las filas para este benchmark (filas sobre las que aplicamos V_g)
    total_rows = D_f.shape[0]
    rows_in_L = list(range(total_rows))
    rows_not_in_L = []


    ## quiero testerar que pasa   
    if formato_test in ('csr', 'csc'):
        tracemalloc.start()
        t0 = time.perf_counter()

        for idx_f, idx_g in zip(index_cycle_columns_f, cycle_columns):
            # Obtener columna de V_g (densa)
            if hasattr(V_g, "tocsc"):
                col_data = V_g[:, idx_g].toarray().ravel()
            else:
                col_data = np.asarray(V_g[:, idx_g]).ravel()

             # Crear una columna dispersa con todos los valores
            n_rows = D_cok.shape[0]
            col_sparse = sp.csc_matrix(
                (col_data, (np.arange(n_rows), np.zeros(n_rows, dtype=int))),
                shape=(n_rows, 1)
            )

        # Asignar toda la columna de una vez (sin bucle)
            D_cok[:, idx_f] = col_sparse

        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = t1 - t0

        # --- PASO 2: reconstruir las asignaciones hechas (sin afectar medición)
        assignments_list = []
        for idx_f, idx_g in zip(index_cycle_columns_f, cycle_columns):
            if hasattr(V_g, "tocsc"):
                col_data = V_g[:, idx_g].toarray().ravel()
            else:
                col_data = np.asarray(V_g[:, idx_g]).ravel()

            for row_idx, val in enumerate(col_data):
                assignments_list.append((int(row_idx), int(idx_f), float(val)))

        total_assignments = len(assignments_list)

        # Clasificar asignaciones: existentes vs nuevas (según snapshot inicial)
        assignments_to_existing = 0
        assignments_to_new = 0
        for row_idx, col_idx, val in assignments_list:
            if (row_idx, col_idx) in existing_coords_before:
                assignments_to_existing += 1
            else:
                assignments_to_new += 1

        # Contar asignaciones que escriben 0
        assignments_by_zero = sum(1 for _, _, val in assignments_list if val == 0)

        # Métricas finales
        nnz_after = D_cok.getnnz()

        return {
            'fmt': formato_test,
            'tiempo_asignaciones_s': float(total_time),
            'memoria_pico_mib': float(peak) / (1024 * 1024),
            'total_asignaciones': int(total_assignments),
            'asignaciones_a_existentes': int(assignments_to_existing),
            'asignaciones_a_nuevas': int(assignments_to_new),
            'asignaciones_por_cero': int(assignments_by_zero),
            'nnz_antes': int(nnz_before),
            'nnz_despues': int(nnz_after),
            'ciclos_procesados': int(len(index_cycle_columns_f)),
            'filas_en_L': int(len(rows_in_L)),
            'filas_no_en_L': int(len(rows_not_in_L)),
        }
    elif formato_test == 'dok':
        tracemalloc.start()
        t0 = time.perf_counter()

        # --- PASO 1: Asignaciones eficientes sobre DOK
        for idx_f, idx_g in zip(index_cycle_columns_f, cycle_columns):
            # Obtener columna de V_g (densa)
            if hasattr(V_g, "tocsc"):
                col_data = V_g[:, idx_g].toarray().ravel()
            else:
                col_data = np.asarray(V_g[:, idx_g]).ravel()

            # Asignar solo valores distintos de cero (DOK es un dict → O(1) por asignación)
            for row_idx, val in enumerate(col_data):
                if val != 0:
                    D_cok[row_idx, idx_f] = val
                elif (row_idx, idx_f) in D_cok:
                    # Si el valor es cero y existía antes, lo eliminamos
                    del D_cok[row_idx, idx_f]

        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = t1 - t0

        # --- PASO 2: reconstruir las asignaciones hechas (sin afectar medición)
        assignments_list = []
        for idx_f, idx_g in zip(index_cycle_columns_f, cycle_columns):
            if hasattr(V_g, "tocsc"):
                col_data = V_g[:, idx_g].toarray().ravel()
            else:
                col_data = np.asarray(V_g[:, idx_g]).ravel()

            for row_idx, val in enumerate(col_data):
                assignments_list.append((int(row_idx), int(idx_f), float(val)))

        total_assignments = len(assignments_list)

        # Clasificar asignaciones: existentes vs nuevas (según snapshot inicial)
        assignments_to_existing = 0
        assignments_to_new = 0
        for row_idx, col_idx, val in assignments_list:
            if (row_idx, col_idx) in existing_coords_before:
                assignments_to_existing += 1
            else:
                assignments_to_new += 1

        # Contar asignaciones que escriben 0
        assignments_by_zero = sum(1 for _, _, val in assignments_list if val == 0)

        # Métricas finales
        nnz_after = D_cok.getnnz()

        return {
            'fmt': formato_test,
            'tiempo_asignaciones_s': float(total_time),
            'memoria_pico_mib': float(peak) / (1024 * 1024),
            'total_asignaciones': int(total_assignments),
            'asignaciones_a_existentes': int(assignments_to_existing),
            'asignaciones_a_nuevas': int(assignments_to_new),
            'asignaciones_por_cero': int(assignments_by_zero),
            'nnz_antes': int(nnz_before),
            'nnz_despues': int(nnz_after),
            'ciclos_procesados': int(len(index_cycle_columns_f)),
            'filas_en_L': int(len(rows_in_L)),
            'filas_no_en_L': int(len(rows_not_in_L)),
        }
    else:
            
    ## quiero testerar que pasa    
    # --- PASO 1: ejecutar for de asignaciones y medir su tiempo (sin contabilizar)
        tracemalloc.start()
        t0 = time.perf_counter()

        for idx_f, idx_g in zip(index_cycle_columns_f, cycle_columns):
            # Obtener columna de V_g (densa)
            if hasattr(V_g, "tocsc"):
                col_data = V_g[:, idx_g].toarray().ravel()
            else:
                col_data = np.asarray(V_g[:, idx_g]).ravel()

            # Asignar en todas las filas (operación que queremos medir)
            for row_idx, val in enumerate(col_data):
                D_cok[row_idx, idx_f] = val

        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = t1 - t0

        # --- PASO 2: reconstruir las asignaciones hechas (sin afectar medición)
        assignments_list = []
        for idx_f, idx_g in zip(index_cycle_columns_f, cycle_columns):
            if hasattr(V_g, "tocsc"):
                col_data = V_g[:, idx_g].toarray().ravel()
            else:
                col_data = np.asarray(V_g[:, idx_g]).ravel()

            for row_idx, val in enumerate(col_data):
                assignments_list.append((int(row_idx), int(idx_f), float(val)))

        total_assignments = len(assignments_list)

        # Clasificar asignaciones: existentes vs nuevas (según snapshot inicial)
        assignments_to_existing = 0
        assignments_to_new = 0
        for row_idx, col_idx, val in assignments_list:
            if (row_idx, col_idx) in existing_coords_before:
                assignments_to_existing += 1
            else:
                assignments_to_new += 1

        # Contar asignaciones que escriben 0
        assignments_by_zero = sum(1 for _, _, val in assignments_list if val == 0)

        # Métricas finales
        nnz_after = D_cok.getnnz()

        return {
            'fmt': formato_test,
            'tiempo_asignaciones_s': float(total_time),
            'memoria_pico_mib': float(peak) / (1024 * 1024),
            'total_asignaciones': int(total_assignments),
            'asignaciones_a_existentes': int(assignments_to_existing),
            'asignaciones_a_nuevas': int(assignments_to_new),
            'asignaciones_por_cero': int(assignments_by_zero),
            'nnz_antes': int(nnz_before),
            'nnz_despues': int(nnz_after),
            'ciclos_procesados': int(len(index_cycle_columns_f)),
            'filas_en_L': int(len(rows_in_L)),
            'filas_no_en_L': int(len(rows_not_in_L)),
        }

def run_single_benchmark_synthetic(n_rows, n_cols, n_nnz_df, formato_test='lil', repeats=20, random_state=42):
    """
    Ejecuta un benchmark con matrices sintéticas random de distinta densidad.
    
    Args:
        n_rows, n_cols: dimensiones de la matriz
        n_nnz_df: cantidad de elementos no nulos en D_f
        formato_test: formato de matriz a probar
        repeats: número de repeticiones
        random_state: seed
    """
    # Construir entradas una sola vez
    D_f, V_g, R_g, mapping_L = build_synthetic_inputs(
        n_rows=n_rows,
        n_cols=n_cols,
        n_nnz_df=n_nnz_df,
        random_state=random_state
    )
    
    resultados = []
    for _ in range(repeats):
        res = benchmark_assignments(D_f, V_g, R_g, mapping_L, formato_test=formato_test)
        resultados.append(res)

    record = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'n_nnz_df': n_nnz_df,
        'density_ratio': n_nnz_df / (n_rows * n_cols) if (n_rows * n_cols) > 0 else 0,
        'fmt': formato_test,
        'nnz_D_f': int(D_f.nnz),
        'nnz_V_g': int(V_g.nnz),
        'tiempo_asignaciones_s': float(np.mean([r['tiempo_asignaciones_s'] for r in resultados])),
        'memoria_pico_mib': float(np.mean([r['memoria_pico_mib'] for r in resultados])),
        'total_asignaciones': resultados[0]['total_asignaciones'],
        'asignaciones_a_existentes': resultados[0]['asignaciones_a_existentes'],
        'asignaciones_a_nuevas': resultados[0]['asignaciones_a_nuevas'],
        'asignaciones_por_cero': resultados[0]['asignaciones_por_cero'],
    }

    return record


def build_synthetic_inputs(n_rows, n_cols, n_nnz_df, random_state=42):
    """
    Construye entradas sintéticas coherentes para el benchmark.

    - D_f: matriz dispersa (n_rows x n_cols) con aproximadamente n_nnz_df no nulos
    - V_g: matriz (n_rows x n_cols_vg) (en formato csc) con columnas que se usarán para asignar
    - R_g: matriz reducida simulada (n_rows x n_cols_vg) en la que algunas columnas son ceros (ciclos)
    - mapping_L: lista de longitud n_cols_vg que mapea cada columna j de R_g/V_g -> columna en D_f

    Se devuelve (D_f (csr), V_g (csc), R_g (csc), mapping_L (list(int))).
    """
    rs = np.random.RandomState(random_state)

    # --- D_f: construir con n_nnz_df valores aleatorios no nulos (valores 1..9)
    total_cells = int(n_rows) * int(n_cols)
    nnz = max(1, min(int(n_nnz_df), total_cells))
    rows = rs.randint(0, n_rows, size=nnz)
    cols = rs.randint(0, n_cols, size=nnz)
    data = rs.randint(1, 10, size=nnz).astype(float)
    D_f = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

    # --- columnas de V_g / R_g (cantidad razonable)
    n_cols_vg = max(1, n_cols // 4)

    # V_g: datos que se escribirán en las columnas target de D_f
    # Usamos formato csc para poder extraer columnas eficientemente con V_g[:, j].toarray()
    density_vg = min(0.2, max(0.01, 50.0 / max(1, n_rows)))
    V_g = sp.random(n_rows, n_cols_vg, density=density_vg, format='csc', random_state=rs,
                    data_rvs=lambda k: rs.randint(0, 3, size=k).astype(float))

    # R_g: simulamos la matriz reducida. Tendrá algunas columnas completamente cero (ciclos)
    R_g = sp.random(n_rows, n_cols_vg, density=max(0.001, min(0.05, 5.0 / max(1, n_rows))),
                    format='lil', random_state=rs, data_rvs=lambda k: rs.randint(0, 3, size=k).astype(float))

    # Hacer cero explícitamente un conjunto de columnas para asegurar ciclos
    n_zero_cols = max(1, n_cols_vg // 3)
    zero_cols = rs.choice(n_cols_vg, size=n_zero_cols, replace=False)
    for j in zero_cols:
        # asignar columna j como todo cero
        R_g[:, j] = 0

    R_g = R_g.tocsc()

    # mapping_L: para cada columna j de R_g/V_g elegimos una columna en D_f donde se escribirá
    if n_cols_vg <= n_cols:
        mapping = rs.choice(n_cols, size=n_cols_vg, replace=False)
    else:
        mapping = rs.choice(n_cols, size=n_cols_vg, replace=True)

    mapping_L = [int(x) for x in mapping]

    return D_f, V_g, R_g, mapping_L



def main():
    """Ejecuta grid de benchmarks con matrices esparsas random de distinta densidad."""
    
    if step1 is None:
        print("Error: No se pudieron importar las funciones necesarias.")
        return
    
    # Configuración fija
    OUT_FILENAME = os.path.join('benchmarks', 'csv_y_vis', 'benchmark_step4_assignments.csv')
    REPEATS = 20
    RANDOM_STATE = 42
    
    os.makedirs(os.path.dirname(OUT_FILENAME) or '.', exist_ok=True)
   
    # === MODO SINTÉTICO: Matrices random con distinta densidad ===
    # Diferentes tamaños de matriz
    matrix_sizes = [(100, 100), (300, 300), (600, 600), (1000, 1000), (1500, 1500), (2000, 2000), (2500, 2500)]
    
    # Diferentes densidades (porcentajes del total de elementos)
    density_percentages = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    formatos_test = ['lil', 'dok', 'csr', 'csc']

    fieldnames = [
        'n_rows', 'n_cols', 'n_nnz_df', 'density_ratio', 'fmt',
        'nnz_D_f', 'nnz_V_g',
        'tiempo_asignaciones_s', 'memoria_pico_mib',
        'total_asignaciones', 'asignaciones_a_existentes', 'asignaciones_a_nuevas',
        'asignaciones_por_cero',
    ]

    with open(OUT_FILENAME, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n_rows, n_cols in matrix_sizes:
            for density_pct in density_percentages:
                # Calcular la cantidad real de elementos no nulos
                n_nnz_df = max(1, int(density_pct * n_rows * n_cols))
                
                for formato_test in formatos_test:
                    print(f"size={n_rows}x{n_cols}, density={density_pct*100:.1f}% (n_nnz={n_nnz_df}), fmt={formato_test}")
                    try:
                        rec = run_single_benchmark_synthetic(
                            n_rows=n_rows,
                            n_cols=n_cols,
                            n_nnz_df=n_nnz_df,
                            formato_test=formato_test,
                            repeats=REPEATS,
                            random_state=RANDOM_STATE,
                        )
                        writer.writerow(rec)
                        f.flush()
                    except Exception as e:
                        print(f"Error en experimento (sintético): {e}")
                        import traceback
                        traceback.print_exc()
                        rec = {k: None for k in fieldnames}
                        rec.update({'n_rows': n_rows, 'n_cols': n_cols, 'n_nnz_df': n_nnz_df, 'fmt': formato_test})
                        writer.writerow(rec)
                        f.flush()

    print(f'\nBenchmark (sintético) completado. Resultados guardados en {OUT_FILENAME}')



if __name__ == '__main__':
    main()
