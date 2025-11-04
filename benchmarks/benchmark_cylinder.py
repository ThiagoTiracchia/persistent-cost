"""
Benchmark que mide tiempos y memoria por paso de la pipeline en
`persistent_cost.cylinder` (cylindermatrix, step1, step2, step3, step4_fast).

Genera entradas sintéticas (matrices dispersas y nubes de puntos) y ejecuta
cada paso por separado midiendo tiempo (perf_counter) y memoria (tracemalloc).

Salida: CSV con una fila por experimento incluyendo tiempos y memoria de cada
paso. El script intenta importar las funciones del módulo `persistent_cost.cylinder`.
Si la importación falla, aborta con un error claro.

Uso:
    python -m benchmarks.benchmark_cylinder --out results.csv --repeats 3
"""

import argparse
import sys
import csv
import os
import time
import tracemalloc
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist


# Intentamos importar las funciones a medir
try:
    from persistent_cost.cylinder import (
        cylindermatrix,
        step1,
        step2,
        step3,
        step4,
        step4_fast,
    )
except Exception as e:
    cylindermatrix = step1 = step2 = step3 = step4_fast = None
    _import_error = e


def random_sparse_matrix(n_rows, n_cols, cant_nonzeros, formato_matriz='csc', random_state=None):
    rs = np.random.RandomState(random_state)
    if cant_nonzeros <= 0:
        coo = sp.coo_matrix((n_rows, n_cols))
    else:
        rows = rs.randint(0, n_rows, size=cant_nonzeros)
        cols = rs.randint(0, n_cols, size=cant_nonzeros)
        data = rs.random_sample(cant_nonzeros)
        coo = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    fmt = formato_matriz.lower()
    if fmt == 'lil':
        return coo.tolil()
    if fmt == 'coo':
        return coo.tocoo()
    if fmt == 'csc':
        return coo.tocsc()
    if fmt == 'dok':
        return coo.todok()
    if fmt == 'csr':
        return coo.tocsr()
    raise ValueError(f"Formato desconocido: {formato_matriz}")


def measure_call(func, repeats, *args, **kwargs):
    #Mide tiempo (mediana) y pico de memoria (mediana) de llamar func(*args).
    times = []
    mems = []
    ret = None
    for _ in range(repeats):
        tracemalloc.stop()
        tracemalloc.start()
        t0 = time.perf_counter()
        ret = func(*args, **kwargs)
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(t1 - t0)
        mems.append(peak)
    return ret, float(np.median(times)), float(np.median(mems) / (1024 * 1024)), times, mems





def run_single_experiment(n_rows, n_cols, formato_matriz, cant_nonzeros, p, repeats=3, random_state=42):
    rs = np.random.RandomState(random_state)

    if cylindermatrix is None:
        raise RuntimeError(f"No se pudieron importar funciones desde persistent_cost.cylinder: {_import_error}")

    # Generamos nubes de puntos para medir cylindermatrix
    nX = max(2, n_rows // 3)
    nY = max(2, n_rows - nX)
    X = rs.random_sample((nX, 2))
    Y = rs.random_sample((nY, 2))
    f_map = rs.randint(0, nY, size=nX)

    # Creamos D_f y D_g en el formato elegido
    D_f = random_sparse_matrix(n_rows, n_cols, cant_nonzeros, formato_matriz=formato_matriz, random_state=rs.randint(0, 2**31))
    D_g = random_sparse_matrix(n_rows, p, max(1, int(0.005 * n_rows * p)), formato_matriz=formato_matriz, random_state=rs.randint(0, 2**31))

    # mapping_L: lista de longitud p con índices destino en D_f
    if p > n_cols:
        mapping_L = [int(x) for x in rs.randint(0, n_cols, size=p)]
    else:
        mapping_L = [int(x) for x in rs.choice(n_cols, size=p, replace=False)]

    record = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'fmt': formato_matriz,
        'nnz_pre': cant_nonzeros,
        'p': p,
      
    }

    # 1) cylindermatrix
    dX = pdist(X)
    dY = pdist(Y)
    _, t_cyl, m_cyl, _, _ = measure_call(cylindermatrix, repeats, dX, dY, f_map)
    record['cylindermatrix_time_s'] = t_cyl
    record['cylindermatrix_mem_mib'] = m_cyl

    # 2) step1
    out, t_s1, m_s1, _, _ = measure_call(step1, repeats, D_g, D_f)
    try:
        R_g, V_g, R_f, V_f = out
    except Exception:
        R_g = V_g = R_f = V_f = None
    record['step1_time_s'] = t_s1
    record['step1_mem_mib'] = m_s1

    # 3) step2
    out, t_s2, m_s2, _, _ = measure_call(step2, repeats, D_f, D_g, mapping_L)
    try:
        R_im, V_im = out
    except Exception:
        R_im = V_im = None
    record['step2_time_s'] = t_s2
    record['step2_mem_mib'] = m_s2

    # 4) step3
    out, t_s3, m_s3, _, _ = measure_call(step3, repeats, R_im, V_im, mapping_L)
    try:
        R_ker, V_ker, cycle_columns_Vim = out
    except Exception:
        R_ker = V_ker = cycle_columns_Vim = None
    record['step3_time_s'] = t_s3
    record['step3_mem_mib'] = m_s3


    # 5) step4 (original) -- medir y capturar error si ocurre
    try:
        _, t_s4, m_s4, _, _ = measure_call(step4, repeats, D_f, V_g, R_g, mapping_L)
        record['step4_time_s'] = t_s4
        record['step4_mem_mib'] = m_s4
        record['step4_error'] = ''
    except Exception as e:
        record['step4_time_s'] = None
        record['step4_mem_mib'] = None
        record['step4_error'] = str(e)

    # 6) step4_fast -- medir también la versión rápida
    try:
        _, t_s4f, m_s4f, _, _ = measure_call(step4_fast, repeats, D_f, V_g, R_g, mapping_L)
        record['step4_fast_time_s'] = t_s4f
        record['step4_fast_mem_mib'] = m_s4f
        record['step4_fast_error'] = ''
    except Exception as e:
        record['step4_fast_time_s'] = None
        record['step4_fast_mem_mib'] = None
        record['step4_fast_error'] = str(e)

    return record


def default_grid():
    sizes = [ (100,100) ,(400, 400), (800, 800) ]
    formats = ['lil', 'coo', 'csc', 'dok', 'csr']
    nnz_pres = [ int(0.01 * s[0] * s[1]) for s in sizes ]
    p = 80
    k_news = [ 10, 40, 80 ]
    return sizes, formats, nnz_pres, p, k_news


def main(argv=None):
    # Use fixed, non-configurable run parameters per user request
    # CSV name, number of repeats and random seed are fixed and cannot be overridden from CLI
    # write results to a fixed subfolder and filename
    OUT_FILENAME = os.path.join('results', 'benchmark.csv')
    REPEATS = 3
    RANDOM_STATE = 42

    sizes, formats, nnz_pres, p, k_news = default_grid()

    out_path = OUT_FILENAME
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # fieldnames include per-step timing/memory
    fieldnames = [
        'n_rows','n_cols','fmt','nnz_pre','p','k_new','use_dense_vg',
        'cylindermatrix_time_s','cylindermatrix_mem_mib',
        'step1_time_s','step1_mem_mib',
        'step2_time_s','step2_mem_mib',
        'step3_time_s','step3_mem_mib',
        'step4_time_s','step4_mem_mib','step4_error',
        'step4_fast_time_s','step4_fast_mem_mib','step4_fast_error',
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (n_rows, n_cols), nnz_pre in zip(sizes, nnz_pres):
            for fmt in formats:
                
                        print(f"Running: size={n_rows}x{n_cols}, fmt={fmt}, nnz_pre={nnz_pre}, p={p}")
                        try:
                            res = run_single_experiment(n_rows, n_cols, fmt, nnz_pre, p, repeats=REPEATS, random_state=RANDOM_STATE)
                        except Exception as e:
                            print(f"Error en experimento: {e}")
                            res = {k: None for k in fieldnames}
                            res.update({'n_rows': n_rows, 'n_cols': n_cols, 'fmt': fmt, 'nnz_pre': nnz_pre, 'p': p})
                        writer.writerow(res)
                        f.flush()

    print('\nBenchmarks completados. Resultados guardados en', out_path)


if __name__ == '__main__':
    main()
