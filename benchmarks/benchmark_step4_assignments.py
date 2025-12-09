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


def _col_dense_from_Vg(V_g, idx_g):
    """Devuelve vector 1D numpy con la columna idx_g de V_g (compatible con sparse/dense V_g)."""
    if hasattr(V_g, "tocsc"):
        return V_g[:, idx_g].toarray().ravel()
    else:
        return np.asarray(V_g[:, idx_g]).ravel()


def _build_full_col_coo(n_rows, n_cols, idx_f, col_data, rows_indices=None):
    """
    Construye una matriz COO del tamaño (n_rows, n_cols) que contiene únicamente la columna idx_f,
    con entradas en rows_indices (o todas si rows_indices is None) donde col_data != 0.
    Se omiten explícitamente los ceros (para que la adición/substracción elimine entradas previas).
    """
    if rows_indices is None:
        rows = np.arange(n_rows, dtype=int)
        vals = np.asarray(col_data)
    else:
        rows = np.asarray(rows_indices, dtype=int)
        vals = np.asarray(col_data)[rows_indices]

    mask = vals != 0
    if mask.sum() == 0:
        # columna vacía
        return sp.coo_matrix((n_rows, n_cols))
    rows_nz = rows[mask]
    vals_nz = vals[mask]
    cols_nz = np.full(len(rows_nz), idx_f, dtype=int)
    return sp.coo_matrix((vals_nz, (rows_nz, cols_nz)), shape=(n_rows, n_cols))


def benchmark_assignments(D_f, V_g, R_g, mapping_L, formato_test='lil'):
    """
    Benchmark del paso de asignaciones del cokernel,
    usando las implementaciones eficientes definitivas.
    """
    from scipy import sparse
    from scipy.sparse import csc_matrix, lil_matrix, dok_matrix
    # ================================
    # MÉTRICAS INICIALES
    # ================================
    nnz_before = D_f.getnnz()

    # Copia según formato
    if formato_test == 'lil':
        D_cok = D_f.tolil()
    elif formato_test == 'step4':
        D_cok = D_f.tolil()
    elif formato_test == 'csc':
        D_cok = D_f.tocsc()
    else:
        raise ValueError("Formato desconocido")
    
    total_rows, total_cols = D_f.shape

    
    cycle_columns = [c for c in range(R_g.shape[1]) if R_g[:, c].getnnz() == 0]
    cycle_columns = [c for c in cycle_columns if c < len(mapping_L)]
    index_cycle_columns_f = [mapping_L[c] for c in cycle_columns]
    rows_in_L = np.array(list(mapping_L)) 
    # rows_not_in_L = np.setdiff1d(np.arange(total_rows), rows_in_L) # Opcional si usamos máscara
    total_rows = D_f.shape[0]
    rows_in_L = [i for i in range(total_rows) if i in mapping_L]
    rows_not_in_L = [i for i in range(total_rows) if i not in rows_in_L]    
    # Snapshot para clasificar asignaciones
    existing_coords_before = set()
    D_f_coo = D_f.tocoo()
    for i, j in zip(D_f_coo.row, D_f_coo.col):
        existing_coords_before.add((int(i), int(j)))

    # A qué columnas de D_f corresponde cada columna ciclo
    index_cycle_columns_f = [mapping_L[c] for c in cycle_columns]

    n_rows = D_f.shape[0]

    # ================================
    #   PASO 1: ASIGNACIONES
    # ================================
    tracemalloc.start()
    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # IMPLEMENTACIÓN 1: step4
    # ------------------------------------------------------------
    if formato_test == "step4":
        for j_idx, (idx_f, idx_g) in enumerate(zip(index_cycle_columns_f, cycle_columns)):        

        # Poner en filas de L la columna correspondiente de V_g
            # Aseguramos que V_g esté en formato compatible (dense o sparse)
            if hasattr(V_g, "tocsc"):
                col_data = V_g[:, idx_g].toarray().ravel()
            else:
                col_data = V_g[:, idx_g]

            for row_idx, val in zip(rows_in_L, col_data):
                D_cok[row_idx, idx_f] = val

            # En filas no L poner cero
            for row_idx in rows_not_in_L:
                D_cok[row_idx, idx_f] = 0

        # Convertir a formato CSR para operaciones eficientes
        D_cok = D_cok.tocsr()
        # Pre-calcular slice para rows_not_in_L es costoso, mejor poner todo a 0 y re-escribir
        # O usar mask booleana
        mask_in_L = np.zeros(total_rows, dtype=bool)
        mask_in_L[rows_in_L] = True

    # ------------------------------------------------------------
    # IMPLEMENTACIÓN 2: DOK (solo escribir nonzeros y borrar ceros)
    # ------------------------------------------------------------
    elif formato_test == "lil":
        mask_in_L = np.zeros(total_rows, dtype=bool)
        mask_in_L[rows_in_L] = True
        rows_not_L_indices = np.where(~mask_in_L)[0]
        for idx_f, idx_g in zip(index_cycle_columns_f, cycle_columns):
        # 1. Obtener columna completa
            if sparse.issparse(V_g):
                col_data = V_g[:, idx_g].toarray().ravel()
            else:
                col_data = np.asarray(V_g[:, idx_g]).ravel()

            # 2. REPLICAR LOGICA step4: Tomar los primeros N elementos
            # donde N es la cantidad de filas en L.
            n_limit = len(rows_in_L)
            
            # Cuidado: zip se detiene en el más corto. 
            limit = min(n_limit, len(col_data))
            
            vals_to_assign = col_data[:limit]     # <--- AQUÍ ESTÁ LA CLAVE (Posicional)
            target_rows = rows_in_L[:limit]       # <--- Filas destino secuenciales

            # Asignación vectorizada
            D_cok[target_rows, idx_f] = vals_to_assign
            
            # Poner ceros en el resto (filas NO en L)
            # Manera rápida en LIL: setear columna a 0 primero? No, perderíamos lo de arriba.
            # Lo hacemos con máscara inversa
            # (Esto es lento en bucle, pero necesario para replicar exactitud si hay basura previa)
            # Una forma rápida:
            # matriz_lil[~mask_in_L, idx_f] = 0  <-- Lento en LIL si mask es grande
            pass 
        
        rows_not_L_indices = np.where(~mask_in_L)[0]
        if len(rows_not_L_indices) > 0:
            # LIL soporta esto decentemente
            for idx_f in index_cycle_columns_f:
                D_cok[rows_not_L_indices, idx_f] = 0

        D_cok = D_cok.tocsc()
    # ------------------------------------------------------------
    # IMPLEMENTACIÓN 3: CSC
    # ------------------------------------------------------------
    elif formato_test == "csc":
      
      for idx_f, idx_g in zip(index_cycle_columns_f, cycle_columns):
        if sparse.issparse(V_g):
            col_data = V_g[:, idx_g].toarray().ravel()
        else:
            col_data = np.asarray(V_g[:, idx_g]).ravel()

        # Lógica step4: Primeros N valores
        limit = min(len(rows_in_L), len(col_data))
        vals = col_data[:limit]
        rows = rows_in_L[:limit]
        
        # Construir columna esparsa nueva
        # Todo lo que no esté en 'rows' será implícitamente 0 (que es lo que step4 hace con rows_not_in_L)
        new_col = csc_matrix((vals, (rows, np.zeros(limit))), shape=(total_rows, 1))
        
        D_cok[:, idx_f] = new_col


    else:
        raise ValueError("Formato inválido")

    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = t1 - t0

    # ================================
    #   PASO 2: Reconstrucción
    # ================================
    assignments_list = []
    for idx_f, idx_g in zip(index_cycle_columns_f, cycle_columns):

        col_data = (
            V_g[:, idx_g].toarray().ravel()
            if hasattr(V_g, "tocsc")
            else np.asarray(V_g[:, idx_g]).ravel()
        )

        for row_idx, val in enumerate(col_data):
            assignments_list.append((int(row_idx), int(idx_f), float(val)))

    total_assignments = len(assignments_list)

    # clasificación
    assignments_to_existing = sum(
        (r, c) in existing_coords_before for r, c, _ in assignments_list
    )
    assignments_to_new = total_assignments - assignments_to_existing
    assignments_by_zero = sum(val == 0 for _, _, val in assignments_list)

    nnz_after = D_cok.getnnz()

    # ================================
    # DEVOLVER RESULTADOS
    # ================================
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
    
    formatos_test = ['lil', 'step4', 'csc']

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
