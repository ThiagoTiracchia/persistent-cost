import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from benchmark_step4_assignments import build_synthetic_inputs
from scipy.sparse import csr_matrix, csc_matrix

def set_sparse_column(d_csr, idx_f, col_data, rows_in_L, total_rows):
    """
    Actualiza la columna idx_f de matriz_posta (CSR) de forma eficiente.
    
    - col_data: vector denso con |rows_in_L| valores
    - rows_in_L: índices de filas donde colocar esos valores
    """

    # 1) Crear columna dispersa SOLO con valores en rows_in_L
    col_sparse = csc_matrix(
        (col_data, rows_in_L, [0, len(rows_in_L)]),
        shape=(total_rows, 1)
    )

    # 2) Asignar toda la columna de una vez (eficiente en CSR)
    d_csr[:, idx_f] = col_sparse



def matrices_iguales(A, B):
    """Compara dos matrices dispersas o densas."""
    Acpy = A.copy()
    Bcpy = B.copy()
   
    if sp.issparse(Acpy):
        Acpy = Acpy.toarray()
    if sp.issparse(Bcpy):
        Bcpy = Bcpy.toarray()
    print("Matriz A:")
    print(Acpy)
    print("Matriz B:")
    print(Bcpy)
    print("¿Son iguales?:")
    print((A != B).nnz == 0)

    diff = (A - B).tocoo()
    print("Diferencias (índices y valores):")
    for i, j, v in zip(diff.row, diff.col, diff.data):
        print(f"  Fila {i}, Columna {j}: {v}")
    return np.array_equal(Acpy, Bcpy)


import numpy as np
import scipy.sparse as sp

import numpy as np
import scipy.sparse as sp




def comparar_matrices(D_f, V_g, R_g, mapping_L):
   
    from scipy import sparse
    from scipy.sparse import csc_matrix, lil_matrix, dok_matrix

    """
    Construye 4 versiones de D_cok (CSC, LIL, DOK, POSTA corregida).
    Devuelve: (matriz_original, matriz_csc, matriz_lil, matriz_dok, matriz_posta)
    """

    # --- Preparaciones básicas ---
    total_rows, total_cols = D_f.shape

    # --- Preparación idéntica ---
    # Detectar ciclos
    # if sparse.issparse(R_g):
    #     cycle_columns = np.where(R_g.getnnz(axis=0) == 0)[0]
    # else:
    #     cycle_columns = np.where(~R_g.any(axis=0))[0]
    cycle_columns = [c for c in range(R_g.shape[1]) if R_g[:, c].getnnz() == 0]
    cycle_columns = [c for c in cycle_columns if c < len(mapping_L)]
    index_cycle_columns_f = [mapping_L[c] for c in cycle_columns]
    
    rows_in_L = np.array(list(mapping_L)) 
    # rows_not_in_L = np.setdiff1d(np.arange(total_rows), rows_in_L) # Opcional si usamos máscara
    total_rows = D_f.shape[0]
    rows_in_L = [i for i in range(total_rows) if i in mapping_L]
    rows_not_in_L = [i for i in range(total_rows) if i not in rows_in_L]
    # Copias
    matriz_lil = D_f.tolil(copy=True)
    matriz_csc = D_f.tocsc(copy=True)
    matriz_step4 =  D_f.tolil(copy=True)

    for j_idx, (idx_f, idx_g) in enumerate(zip(index_cycle_columns_f, cycle_columns)):        

        # Poner en filas de L la columna correspondiente de V_g
        # Aseguramos que V_g esté en formato compatible (dense o sparse)
        if hasattr(V_g, "tocsc"):
            col_data = V_g[:, idx_g].toarray().ravel()
        else:
            col_data = V_g[:, idx_g]

        for row_idx, val in zip(rows_in_L, col_data):
            matriz_step4[row_idx, idx_f] = val

        # En filas no L poner cero
        for row_idx in rows_not_in_L:
            matriz_step4[row_idx, idx_f] = 0

    # Convertir a formato CSR para operaciones eficientes
    matriz_step4 = matriz_step4.tocsr()
    # Pre-calcular slice para rows_not_in_L es costoso, mejor poner todo a 0 y re-escribir
    # O usar mask booleana
    mask_in_L = np.zeros(total_rows, dtype=bool)
    mask_in_L[rows_in_L] = True
    
    # ==========================================
    # LIL OPTIMIZADO (Lógica step4 replicada)
    # ==========================================
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
        matriz_lil[target_rows, idx_f] = vals_to_assign
        
        # Poner ceros en el resto (filas NO en L)
        # Manera rápida en LIL: setear columna a 0 primero? No, perderíamos lo de arriba.
        # Lo hacemos con máscara inversa
        # (Esto es lento en bucle, pero necesario para replicar exactitud si hay basura previa)
        # Una forma rápida:
        # matriz_lil[~mask_in_L, idx_f] = 0  <-- Lento en LIL si mask es grande
        pass 

    # Corrección de ceros fuera de L (vectorizada fuera del bucle de ser posible, o dentro)
    # Para exactitud estricta con step4:
    rows_not_L_indices = np.where(~mask_in_L)[0]
    if len(rows_not_L_indices) > 0:
        # LIL soporta esto decentemente
        for idx_f in index_cycle_columns_f:
             matriz_lil[rows_not_L_indices, idx_f] = 0

    matriz_lil = matriz_lil.tocsc()

    # ==========================================
    # CSC OPTIMIZADO (Lógica step4 replicada)
    # ==========================================
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
        
        matriz_csc[:, idx_f] = new_col

    return matriz_lil, matriz_csc, matriz_step4



def test_step4_all_implementations():
    # Construir matrices sintéticas pequeñas
    D_f, V_g, R_g, mapping_L = build_synthetic_inputs(
        n_rows=10,
        n_cols=10,
        n_nnz_df=80,
        random_state=123
    )

    matriz_lil, matriz_csc, matriz_step4 = comparar_matrices(
        D_f, V_g, R_g, mapping_L
    )

    # Asegurar que todas coinciden con la versión "posta"
    print((matriz_csc != matriz_lil).nnz)  # debe ser 0
    print((matriz_step4 != matriz_lil).nnz)  # debe ser 0
    print((matriz_step4 != matriz_csc).nnz)  # debe ser 0

    assert matrices_iguales(matriz_csc, matriz_lil)
    assert matrices_iguales(matriz_step4, matriz_lil)

 
if __name__ == "__main__":
    
    test_step4_all_implementations()
    print("Todos los tests pasaron correctamente.")



