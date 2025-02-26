"""
Código del método multicriterio AHP
Andrés Felipe Martínez Rodríguez
"""

# Importación de bibliotecas necesarias
import numpy as np
import pandas as pd

# Funciones para obtener pesos y normalizar cualquier matriz

def matriz_comparacion(vector_importancia):
    """
    Genera una matriz de comparación a partir de un vector de importancia.

    Entradas:
        vector_importancia (list): Vector de porcentajes de importancia de los criterios.

    Salidas:
        Matriz_comparacion_criterios (DataFrame): Matriz de comparación de criterios.
    """
    # Convertir el vector a un array de numpy
    vector_importancia = np.array(vector_importancia)
    A = vector_importancia / 10
    n = len(vector_importancia)

    # Inicializar matriz de comparación
    matriz_comparacion = np.zeros((n, n))

    # Llenar la matriz de comparación
    for i in range(n):
        for j in range(n):
            if i == j:
                matriz_comparacion[i, i] = 1  # Diagonal principal es 1
            elif i > j:
                matriz_comparacion[i, j] = A[i] / A[j]
                matriz_comparacion[j, i] = 1 / matriz_comparacion[i, j]
            else:
                matriz_comparacion[i, j] = A[i] / A[j]
                matriz_comparacion[j, i] = 1 / matriz_comparacion[i, j]

    # Crear nombres de filas y columnas
    filas = [f'Criterio{i + 1}' for i in range(n)]
    columnas = [f'Criterio{i + 1}' for i in range(n)]

    # Convertir la matriz a un DataFrame de pandas
    Matriz_comparacion_criterios = pd.DataFrame(matriz_comparacion, index=filas, columns=columnas)

    return Matriz_comparacion_criterios


def normalizar_matriz(matriz):
    """
    Normaliza una matriz dada.

    Entradas:
        matriz (DataFrame o numpy array): Matriz a normalizar.

    Salidas:
        matriz_normalizada (DataFrame o numpy array): Matriz normalizada.
    """
    # Sumar los elementos de cada columna
    suma_columnas = matriz.sum(axis=0)
    
    # Dividir cada elemento de la matriz por la suma de su columna
    matriz_normalizada = matriz / suma_columnas
    
    return matriz_normalizada


def pesos(matriz_normalizada):
    """
    Calcula los pesos a partir de una matriz normalizada.

    Entradas:
        matriz_normalizada (DataFrame o numpy array): Matriz normalizada.

    Salidas:
        vector_pesos (numpy array): Vector de pesos.
    """
    # Calcular el promedio de cada fila
    vector_pesos = matriz_normalizada.mean(axis=1)

    return vector_pesos


def verificar_consistencia(matriz_comparacion, vector_pesos):
    """
    Verifica la consistencia de la matriz de comparación.

    Entradas:
        matriz_comparacion (numpy array): Matriz de comparación de criterios.
        vector_pesos (numpy array): Vector de pesos (prioridades).

    Salidas:
        CR (float): Razón de consistencia.
        es_consistente (bool): True si la matriz es consistente (CR < 0.10), False en caso contrario.
    """
    # Paso 1: Calcular el vector de consistencia
    vector_consistencia = np.dot(matriz_comparacion, vector_pesos)

    # Paso 2: Calcular el valor máximo (λ_max)
    lambda_max = np.mean(vector_consistencia / vector_pesos)

    # Paso 3: Calcular el índice de consistencia (CI)
    n = len(vector_pesos)
    CI = (lambda_max - n) / (n - 1)

    # Paso 4: Obtener el índice de consistencia aleatoria (RI)
    # Tabla de RI para diferentes valores de n
    RI_table = {
        1: 0.0,
        2: 0.0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }
    RI = RI_table.get(n, 1.49)  # Si n > 10, se usa 1.49 como valor por defecto

    # Paso 5: Calcular la razón de consistencia (CR)
    CR = CI / RI

    # Paso 6: Determinar si la matriz es consistente
    es_consistente = CR < 0.10

    # Mostrar resultados y advertencias
    print(f"Razón de consistencia (CR): {CR}")
    print(f"¿Es consistente? {es_consistente}")

    if not es_consistente:
        print("Advertencia: La matriz de comparación no es consistente. Revise las comparaciones.")

    return CR, es_consistente


# Función principal AHP

def AHP(base_normalizada, vector_pesos):
    """
    Realiza el proceso de Analytic Hierarchy Process (AHP) y devuelve los resultados.

    Entradas:
        base_normalizada (DataFrame o numpy array): Base de datos normalizada.
        vector_pesos (numpy array): Vector de pesos.

    Salidas:
        ranking (DataFrame): Ranking de activos, de mayor a menor.
    """
    # Convertir a arrays de numpy
    vector_pesos = np.array(vector_pesos)
    base_normalizada = np.array(base_normalizada)

    # Verificar que el número de columnas coincida con el número de pesos
    if base_normalizada.shape[1] != len(vector_pesos):
        raise ValueError("El número de columnas de la base de datos es diferente al número de elementos del vector de pesos.")
    
    # Multiplicar la base normalizada por los pesos
    matriz_con_pesos = base_normalizada * vector_pesos

    # Sumar las filas (axis=1 para sumar a lo largo de las columnas)
    suma_filas = np.sum(matriz_con_pesos, axis=1)

    # Crear DataFrame para el ranking
    ranking = pd.DataFrame(suma_filas, columns=["RANKING"])

    # Agregar columna de activos
    ranking['Activos'] = range(1, len(suma_filas) + 1)

    # Reordenar columnas
    ranking = ranking[['Activos', 'RANKING']]

    return ranking


#Ejecutar funciones
# Vector de importancia en porcentajes para cada criterio
vector_importancia = [10, 20, 10, 15, 25, 15, 5]

# Generar matriz de comparación a partir del vector de importancia
matriz_comparacion_criterios = matriz_comparacion(vector_importancia)

# Normalizar la matriz de comparación
matriz_comparacion_normalizada = normalizar_matriz(matriz_comparacion_criterios)

# Calcular los pesos de los criterios a partir de la matriz normalizada
pesos_criterios = pesos(matriz_comparacion_normalizada)

# Verificar la consistencia de la matriz de comparación
es_consistente = verificar_consistencia(matriz_comparacion_criterios, pesos_criterios)
print("¿La matriz es consistente?:", es_consistente)
print('Pesos de los criterios:', pesos_criterios)

# URL del archivo de base de datos
ruta_archivo = 'Base de datos.xlsm'

# Leer el archivo de Excel
datos = pd.read_excel(ruta_archivo, sheet_name='Base de datos')

# Extraer columnas de interés para el estudio
columnas_estudio = ['Saidi ', 'VURR [años]', 'Costo de reposición [COP]', 'PCB', 'ENS[$]', 'Criticidad', 'Acceso al activo']
datos_extraidos = datos[columnas_estudio]
print("Datos a estudiar:\n", datos_extraidos)

# Normalizar los datos extraídos
datos_normalizados = normalizar_matriz(datos_extraidos)
print("Datos normalizados:\n", datos_normalizados)

# Aplicar el proceso AHP para obtener los resultados
resultados_ahp = AHP(datos_normalizados, pesos_criterios)

# Ordenar los resultados por ranking de manera descendente
resultados_ahp_ordenados = resultados_ahp.sort_values(by="RANKING", ascending=False)
print("Resultados del AHP:\n", resultados_ahp_ordenados)

# Guardar los resultados en un archivo CSV
resultados_ahp_ordenados.to_csv("Resultados AHP.csv", index=False)



