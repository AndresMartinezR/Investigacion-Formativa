"""
Código del método multicriterio AHP y la asignación de tareas a un determinado número de cuadrillas en un número de días,
considerando las jornadas laborales.

Andrés Felipe Martínez Rodríguez
CC 1110592058
"""

# Importación de bibliotecas necesarias
import numpy as np
import pandas as pd
import heapq


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
    ranking['Obras'] = range(1, len(suma_filas) + 1)

    # Reordenar columnas
    ranking = ranking[['Obras', 'RANKING']]

    #Agregar Columnas de horas de reposicion 


    return ranking

def horas_reposicion(ranking, datos):
    """
    Funcion para agregar la columna de horas de reposicion del activo
    Entrada: Ranking y datos base

    Salida: Agregar columna de tiempo de reposicion y despues de eso se hace el ranking, su salida es es ranking ya con las horas
    """
    # Definir la columna que queremos extraer
    Columna_horas = ['Tiempo de reposición [h]']  # Cerramos correctamente la lista
    
    # Extraer la columna de datos_horas
    datos_horas = datos[Columna_horas]

    # Resetear índices para evitar problemas en la fusión
    datos_horas = datos_horas.reset_index(drop=True)
    ranking = ranking.reset_index(drop=True)

    # Agregar datos_horas como primera columna en ranking
    ranking.insert(0, 'Tiempo de reposición [h]', datos_horas)
    #ordenar de mayor a menor en funcion de los datos de la columna Rankig
    
    ranking_ordenado = ranking.sort_values(by="RANKING", ascending=False)

    return ranking_ordenado

#Asignacion de tareas
def Asignar_cuadrillas(datos: pd.DataFrame, numero_cuadrillas_disponibles: int) -> list:
    """
    Asigna las obras a las cuadrillas disponibles según la priorización de RANKING.

    :param datos: DataFrame con las obras y su priorización.
    :param numero_cuadrillas_disponibles: Número de cuadrillas disponibles para asignación.
    :return: Lista de listas con las obras asignadas a cada cuadrilla.
    """
    # Ordenar los datos por RANKING
    datos_ordenados = datos.sort_values(by="RANKING", ascending=False).reset_index(drop=True)

    # Crear listas vacías para cada cuadrilla
    cuadrillas = [{"obras": [], "tiempos": 0} for _ in range(numero_cuadrillas_disponibles)]
    heap = [(0, i) for i in range(numero_cuadrillas_disponibles)]  # Tupla de tiempo acumulado, cuadrillas 

    # Asignar obras a cuadrillas
    for i in range(len(datos_ordenados)):
        obra = float(datos_ordenados.iloc[i]["Obras"])  # Convertir a float
        tiempo = float(datos_ordenados.iloc[i]["Tiempo de reposición [h]"])  # Convertir a float

        # Obtener la cuadrilla con el menor tiempo acumulado
        total_time, cuadrilla = heapq.heappop(heap)

        # Asignar la obra a esta cuadrilla
        cuadrillas[cuadrilla]["obras"].append((obra, tiempo))
        cuadrillas[cuadrilla]["tiempos"] = total_time + tiempo
        # Reinsertar la cuadrilla en el heap
        heapq.heappush(heap, (cuadrillas[cuadrilla]["tiempos"], cuadrilla)) #Obras(Cuadrilla, tiempo), Cuadrilla

    return cuadrillas

#Asigacion de tareas considerando 8 horas labroales de Lunes a Viernes y sabado solo 4 horas
def Asignar_cuadrillas_diarias(datos: pd.DataFrame, numero_cuadrillas_disponibles: int, num_dias: int) -> list:
    """
    Asigna las obras a las cuadrillas disponibles según la priorización de RANKING,
    considerando que cada sábado (día 6, 12, 18, ...) solo se trabajan 4 horas.

    Entrada: Datos, numero de cuadrillas disponibles, numero de dias disponibles
    Salida: lista de cuadrillas
    """

    # Ordenar las obras por prioridad de mayor a menor (Ranking descendente)
    datos_ordenados = datos.sort_values(by="RANKING", ascending=False).reset_index(drop=True)

    # Inicializar la estructura de cuadrillas:
    # Cada cuadrilla tendrá una lista de días y cada día contendrá un diccionario con "obras" y "tiempos"
    cuadrillas = [[{"obras": [], "tiempos": 0} for _ in range(num_dias)] for _ in range(numero_cuadrillas_disponibles)]

    # Iterar sobre cada obra para asignarla a una cuadrilla y un día disponibles
    for i in range(len(datos_ordenados)):
        obra = float(datos_ordenados.iloc[i]["Obras"])  # Identificador de la obra
        tiempo = float(datos_ordenados.iloc[i]["Tiempo de reposición [h]"])  # Tiempo necesario para la obra
        
        asignada = False  # Bandera que indica si la obra ha sido asignada exitosamente

        # Intentar asignar la obra a una cuadrilla en un día disponible
        for dia in range(num_dias):
            max_horas = 4 if ((dia + 1) % 6 == 0) else 8  # Definir el límite de horas según el día (sábado o no)

            for cuadrilla in range(numero_cuadrillas_disponibles):
                # Verificar si la cuadrilla tiene suficiente espacio para agregar la obra en este día
                if cuadrillas[cuadrilla][dia]["tiempos"] + tiempo <= max_horas:
                    cuadrillas[cuadrilla][dia]["obras"].append((obra, tiempo))  # Agregar la obra
                    cuadrillas[cuadrilla][dia]["tiempos"] += tiempo  # Actualizar el total de horas ocupadas
                    asignada = True  # Marcar la obra como asignada
                    break  # Salir del bucle de cuadrillas ya que la obra fue asignada
            
            if asignada:
                break  # Salir del bucle de días si la obra ya fue asignada
        
        # Si no se pudo asignar en los días disponibles, agregar más días y asignar la obra allí
        if not asignada:
            num_dias += 1  # Aumentar el número de días disponibles
            for cuadrilla in range(numero_cuadrillas_disponibles):
                cuadrillas[cuadrilla].append({"obras": [], "tiempos": 0})  # Agregar un nuevo día vacío para cada cuadrilla
            
            # Asignar la obra al primer día nuevo creado, en la primera cuadrilla disponible
            cuadrillas[0][num_dias - 1]["obras"].append((obra, tiempo))
            cuadrillas[0][num_dias - 1]["tiempos"] += tiempo  # Actualizar el total de horas ocupadas en el nuevo día

    return cuadrillas

# Función para mostrar la asignación de una manera legible
def mostrar_asignacion(cuadrillas):
    """
    Muestra la asignación de obras a cada cuadrilla, incluyendo el tiempo total trabajado.
    
    Entrada: cuadrillas: Lista de cuadrillas con sus respectivas asignaciones de obras y tiempos.

    Salida: Impresión en consola de la distribución de obras y tiempos por cuadrilla.
    """
    for indice, cuadrilla in enumerate(cuadrillas):
        print(f"Cuadrilla {indice + 1}:")
        for obra, tiempo in cuadrilla["obras"]:
            print(f"  Obra: {obra}, Tiempo: {tiempo} horas")
        print(f"  Tiempo total: {cuadrilla['tiempos']} horas")
        print()

# Función para formatear el cronograma de asignación de obras en un DataFrame de Pandas
def organizar_cronograma(cuadrillas):
    """
    Convierte la asignación de cuadrillas en un DataFrame organizado, detallando las obras asignadas 
    por día y cuadrilla.

    Entrada: cuadrillas: Lista de listas que representan las cuadrillas con sus asignaciones de obras y tiempos.

    Salida: DataFrame de Pandas 
    """

    # Lista para almacenar la información estructurada de cada asignación
    cronograma = []

    # Recorrer los días en el cronograma (suponiendo que todas las cuadrillas tienen el mismo número de días)
    for dia in range(len(cuadrillas[0])):  
        
        # Recorrer cada cuadrilla disponible
        for cuadrilla in range(len(cuadrillas)):  
            
            # Recorrer cada obra asignada a la cuadrilla en ese día
            for obra, tiempo in cuadrillas[cuadrilla][dia]["obras"]:
                
                # Agregar la información estructurada a la lista de cronograma
                cronograma.append({
                    "Día": dia + 1,  # Se usa dia + 1 para que inicie en 1 en lugar de 0
                    "Cuadrilla": cuadrilla + 1,  # Se usa cuadrilla + 1 por el mismo motivo
                    "Obra": obra,
                    "Tiempo": tiempo
                })
    
    # Convertir la lista de diccionarios en un DataFrame de Pandas para una mejor presentación
    cronograma_completo = pd.DataFrame(cronograma)
    
    return cronograma_completo  # Se devuelve el DataFrame con el cronograma formateado


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
#print("¿La matriz es consistente?:", es_consistente)
#print('Pesos de los criterios:', pesos_criterios)

# URL del archivo de base de datos
ruta_archivo = 'Base de datos.xlsm'

# Leer el archivo de Excel
datos = pd.read_excel(ruta_archivo, sheet_name='Base de datos')

# Extraer columnas de interés para el estudio
columnas_estudio = ['Saidi ', 'VURR [años]', 'Costo de reposición [COP]', 'PCB', 'ENS[$]', 'Criticidad', 'Acceso al activo']
datos_extraidos = datos[columnas_estudio]
#print("Datos a estudiar:\n", datos_extraidos)

# Normalizar los datos extraídos
datos_normalizados = normalizar_matriz(datos_extraidos)
#print("Datos normalizados:\n", datos_normalizados)

# Aplicar el proceso AHP para obtener los resultados
resultados_ahp = AHP(datos_normalizados, pesos_criterios)

ranking = horas_reposicion(resultados_ahp, datos)

#asignacion = Asignar_cuadrillas(ranking, 10)
#mostrar = mostrar_asignacion(asignacion)

Asignacion_jornada = Asignar_cuadrillas_diarias(ranking, 10, 60)
organizar_asignacion = organizar_cronograma(Asignacion_jornada)
print("Asignacion:", organizar_asignacion)
organizar_asignacion.to_csv("asignacion_cronograma.csv", index=False)



