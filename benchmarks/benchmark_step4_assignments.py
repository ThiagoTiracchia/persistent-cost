import csv
import os
import time
import tracemalloc
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform

try:
    from persistent_cost.cylinder import step1, cylindermatrix
    from persistent_cost.utils.utils import build_ordered_boundary_matrix, match_simplices
    from benchmark_cylinder import random_sparse_matrix
    from persistent_cost.algorithms.sparse_fast import do_pivot_cython as do_pivot
except Exception as e:
    print(f"Error importando: {e}")
    step1 = None
    do_pivot = None


def benchmark_assignments(D_f, V_g, R_g, mapping_L, formato_test='lil'):
    """
    Benchmarkea específicamente las asignaciones elemento por elemento de step4.
    
    Args:
        D_f: Matriz dispersa del complejo K
        V_g: Matriz de transformación de D_g
        R_g: Matriz reducida de D_g
        mapping_L: Lista de índices correspondientes a L
        formato_test: Formato de matriz a usar para D_cok ('lil', 'dok', 'csr', 'csc')
    
    Returns:
        dict con métricas del benchmark
    """
    
    # Preparar datos (igual que en step4)
    total_rows = D_f.shape[0]
    rows_in_L = [i for i in range(total_rows) if i in mapping_L]
    rows_not_in_L = [i for i in range(total_rows) if i not in rows_in_L]
    
    # Columnas de R_g que son ciclos
    cycle_columns = [c for c in range(R_g.shape[1]) if R_g[:, c].getnnz() == 0]
    index_cycle_columns_f = [mapping_L[c] for c in cycle_columns]
    
    # Convertir D_f al formato de test
    if formato_test == 'lil':
        D_cok = D_f.tolil()
    elif formato_test == 'dok':
        D_cok = D_f.todok()
    elif formato_test == 'csr':
        D_cok = D_f.tocsr()
    elif formato_test == 'csc':
        D_cok = D_f.tocsc()
    else:
        D_cok = D_f.tolil()
    
    # Métricas iniciales
    nnz_before = D_cok.getnnz()
    
    # Preparar para medir asignaciones SOLO del bucle:
    # for row_idx, val in zip(rows_in_L, col_data):
    #     D_cok[row_idx, idx_f] = val
    total_assignments = 0
    assignments_to_existing = 0
    assignments_to_new = 0
    assignments_by_zero = 0
    
    # Para detectar coordenadas existentes (solo para formatos que lo soporten eficientemente)
    if formato_test in ['lil', 'dok']:
        if formato_test == 'lil':
            existing_coords = set()
            for i in range(D_cok.shape[0]):
                for j, val in zip(D_cok.rows[i], D_cok.data[i]):
                    existing_coords.add((i, j))
        else:  # dok
            existing_coords = set(D_cok.keys())
    else:
        existing_coords = None
    
    # Medición de tiempo y memoria exclusivamente del bucle de asignaciones sobre filas en L
    total_time = 0.0
    tracemalloc.start()
    peak = 0

    for j_idx, (idx_f, idx_g) in enumerate(zip(index_cycle_columns_f, cycle_columns)):
        # Obtener columna de V_g (fuera del reloj para medir solo el setitem)
        if hasattr(V_g, "tocsc"):
            col_data = V_g[:, idx_g].toarray().ravel()
        else:
            col_data = V_g[:, idx_g]

        # Asignaciones en filas de L (único bloque que medimos)
        t0 = time.perf_counter()
        for row_idx, val in zip(rows_in_L, col_data):
            total_assignments += 1

            # Detectar si es coordenada existente
            if existing_coords is not None:
                if (row_idx, idx_f) in existing_coords:
                    assignments_to_existing += 1
                else:
                    assignments_to_new += 1

            # Detectar si asignamos cero
            if val == 0:
                assignments_by_zero += 1

            # LA ASIGNACIÓN QUE QUEREMOS MEDIR
            D_cok[row_idx, idx_f] = val
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        _, p = tracemalloc.get_traced_memory()
        peak = max(peak, p)

    current, peak_final = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak = max(peak, peak_final)

    # Métricas finales
    nnz_after = D_cok.getnnz()
    
    return {
        'fmt': formato_test,
    'tiempo_asignaciones_s': total_time,
    'memoria_pico_mib': peak / (1024 * 1024),
        'total_asignaciones': total_assignments,
        'asignaciones_a_existentes': assignments_to_existing if existing_coords else None,
        'asignaciones_a_nuevas': assignments_to_new if existing_coords else None,
    'asignaciones_por_cero': assignments_by_zero,
        'nnz_antes': nnz_before,
        'nnz_despues': nnz_after,
        'ciclos_procesados': len(cycle_columns),
        'filas_en_L': len(rows_in_L),
    'filas_no_en_L': len(rows_not_in_L),
    }




def build_realistic_inputs(nX, nY, threshold=0.5, maxdim=1, random_state=42):
    """Construye D_f, D_g y mapping_L de forma realista como en cylinder.py.

    - Genera puntos X, Y aleatorios en [0,1]^2
    - Genera un mapa f: X -> Y aleatorio
    - Construye matrices de distancias dX, dY
    - Construye PX (para X) y Pcyl (para cilindro) con build_ordered_boundary_matrix
    - mapping_L se obtiene con match_simplices(f, simplices_X, simplices_cyl)
    - step1 para obtener R_g, V_g
    """
    rs = np.random.RandomState(random_state)
    X = rs.random_sample((nX, 2))
    Y = rs.random_sample((nY, 2))
    f_map = rs.randint(0, nY, size=nX)

    dX = pdist(X)
    dY = pdist(Y)

    PX, simplices_X = build_ordered_boundary_matrix(
        distance_matrix=squareform(dX), threshold=threshold, maxdim=maxdim
    )

    cyl = cylindermatrix(dX, dY, f_map)
    Pcyl, simplices_cyl = build_ordered_boundary_matrix(
        distance_matrix=cyl, threshold=threshold, maxdim=maxdim
    )

    mapping_L = [int(x) for x in match_simplices(f_map, simplices_X, simplices_cyl)]

    D_g = PX
    D_f = Pcyl
    R_g, V_g, R_f, V_f = step1(D_g, D_f)

    meta = {
        'nX': nX,
        'nY': nY,
        'threshold': threshold,
        'maxdim': maxdim,
        'D_f_rows': int(D_f.shape[0]),
        'D_f_cols': int(D_f.shape[1]),
        'D_f_nnz': int(D_f.nnz),
        'D_g_rows': int(D_g.shape[0]),
        'D_g_cols': int(D_g.shape[1]),
        'D_g_nnz': int(D_g.nnz),
        'L_size': int(len(mapping_L)),
    }

    return D_f, V_g, R_g, mapping_L, meta


def run_single_benchmark_realistic(nX, nY, formato_test='lil', repeats=3, random_state=42, threshold=0.5, maxdim=1):
    
    # Construimos entradas realistas una sola vez
    D_f, V_g, R_g, mapping_L, meta = build_realistic_inputs(
        nX=nX, nY=nY, threshold=threshold, maxdim=maxdim, random_state=random_state
    )

    resultados = []
    for _ in range(repeats):
        res = benchmark_assignments(D_f, V_g, R_g, mapping_L, formato_test=formato_test)
        resultados.append(res)

    record = {
        'nX': nX,
        'nY': nY,
        'fmt': formato_test,
        'threshold': threshold,
        'maxdim': maxdim,
        'D_f_rows': meta['D_f_rows'],
        'D_f_cols': meta['D_f_cols'],
        'D_f_nnz': meta['D_f_nnz'],
        'D_g_rows': meta['D_g_rows'],
        'D_g_cols': meta['D_g_cols'],
        'D_g_nnz': meta['D_g_nnz'],
        'L_size': meta['L_size'],
        'tiempo_asignaciones_s': float(np.median([r['tiempo_asignaciones_s'] for r in resultados])),
        'memoria_pico_mib': float(np.median([r['memoria_pico_mib'] for r in resultados])),
        'total_asignaciones': resultados[0]['total_asignaciones'],
        'asignaciones_a_existentes': resultados[0]['asignaciones_a_existentes'],
        'asignaciones_a_nuevas': resultados[0]['asignaciones_a_nuevas'],
        'asignaciones_por_cero': resultados[0]['asignaciones_por_cero'],
    }

    return record


def main():
    """Ejecuta grid de benchmarks."""
    
    if step1 is None:
        print("Error: No se pudieron importar las funciones necesarias.")
        return
    
    # Configuración fija
    #OUT_FILENAME = os.path.join('benchmarks', 'csv_y_vis', 'benchmark_step4_assignments.csv')
    OUT_FILENAME = os.path.join('benchmarks', 'csv_y_vis', 'benchmark_step4_assignments_realistic.csv')
    REPEATS = 1
    RANDOM_STATE = 42
    
    
    os.makedirs(os.path.dirname(OUT_FILENAME) or '.', exist_ok=True)
   
    # === MODO REALISTA (simple) ===
    # tamaños de puntos moderados para evitar explosión combinatoria
    realistic_sizes = [(30, 30), (60, 60), (90, 90), (100,100) ,(400, 400), (800, 800) ]
    formatos_test_real = ['lil', 'dok', 'csr', 'csc']
    threshold = 0.5
    maxdim = 1

    fieldnames_real = [
        'nX', 'nY', 'fmt', 'threshold', 'maxdim',
        'D_f_rows', 'D_f_cols', 'D_f_nnz',
        'D_g_rows', 'D_g_cols', 'D_g_nnz',
        'L_size',
        'tiempo_asignaciones_s', 'memoria_pico_mib',
        'total_asignaciones', 'asignaciones_a_existentes', 'asignaciones_a_nuevas',
        'asignaciones_por_cero',
    ]

    with open(OUT_FILENAME, 'w', newline='') as f2:
        writer2 = csv.DictWriter(f2, fieldnames=fieldnames_real)
        writer2.writeheader()

        for nX, nY in realistic_sizes:
            for formato_test in formatos_test_real:
                print(f"[Realista] size={nX}x{nY}, fmt={formato_test}, threshold={threshold}, maxdim={maxdim}")
                try:
                    rec = run_single_benchmark_realistic(
                        nX, nY,
                        formato_test=formato_test,
                        repeats=REPEATS,
                        random_state=RANDOM_STATE,
                        threshold=threshold,
                        maxdim=maxdim,
                    )
                    writer2.writerow(rec)
                    f2.flush()
                except Exception as e:
                    print(f"Error en experimento (realista): {e}")
                    import traceback
                    traceback.print_exc()
                    rec = {k: None for k in fieldnames_real}
                    rec.update({'nX': nX, 'nY': nY, 'fmt': formato_test, 'threshold': threshold, 'maxdim': maxdim})
                    writer2.writerow(rec)
                    f2.flush()

    print(f'\nBenchmark (realista) completado. Resultados guardados en {OUT_FILENAME}')


if __name__ == '__main__':
    main()
