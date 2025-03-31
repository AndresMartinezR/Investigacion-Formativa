# Reposición de Activos Eléctricos

## Introducción

La reposición de activos eléctricos es una tarea crítica en la gestión de redes de distribución de energía eléctrica. A medida que los activos envejecen, es necesario determinar qué equipos deben ser reemplazados para mantener la fiabilidad y eficiencia del sistema eléctrico. Este proceso involucra tomar decisiones basadas en múltiples criterios, como el estado de los activos, los costos de mantenimiento, la criticidad de los equipos en la red y otros factores operativos.

Este repositorio se centra en el desarrollo de un modelo para optimizar la reposición de activos eléctricos utilizando dos metodologías clave: **Análisis Jerárquico de Procesos (AHP)** para priorizar activos y **Búsqueda Local Iterativa (ILS)** para asignar las tareas de reposición a cuadrillas de trabajo.

## Descripción

El objetivo principal de este repositorio es proporcionar una herramienta que ayude a optimizar la reposición de activos eléctricos en redes de distribución, teniendo en cuenta tanto la priorización de activos como la asignación eficiente de tareas a cuadrillas de trabajo. El modelo sigue las siguientes fases:

1. **Priorización de activos utilizando AHP**: El Análisis Jerárquico de Procesos (AHP) es un método multicriterio que ayuda a priorizar los activos según múltiples factores. Cada activo se evalúa en función de su importancia en la red, el costo de mantenimiento, su antigüedad, entre otros. La prioridad obtenida a partir de AHP define qué activos deben ser reemplazados primero.

2. **Asignación de tareas mediante ILS**: Una vez priorizados los activos, el algoritmo de Búsqueda Local Iterativa (ILS) se utiliza para asignar las tareas de reposición a las cuadrillas disponibles. El algoritmo ILS es un método de optimización iterativa que busca una asignación eficiente de tareas, minimizando el tiempo total de ejecución o maximizando el uso eficiente de los recursos disponibles, como el número de cuadrillas y el tiempo de trabajo.

## Objetivo

El objetivo de este proyecto es proporcionar una solución integral para la reposición de activos eléctricos en redes de distribución. A través de la combinación de AHP para la priorización de activos y el algoritmo ILS para la asignación eficiente de tareas a cuadrillas, el modelo ayuda a optimizar el proceso de reposición, maximizando la eficiencia operativa y reduciendo los costos.

## Metodología

### 1. **Análisis Jerárquico de Procesos (AHP)**

El método AHP permite estructurar el problema de la reposición de activos evaluando cada activo en función de varios criterios. A través de comparaciones entre activos y criterios, el método asigna un peso a cada activo, lo que ayuda a priorizar los que deben ser reemplazados primero. Este enfoque asegura que las decisiones de reposición estén basadas en una evaluación objetiva y estructurada.

### 2. **Búsqueda Local Iterativa (ILS) para asignación de tareas**

El algoritmo de Búsqueda Local Iterativa (ILS) se aplica a la asignación de las tareas de reposición a las cuadrillas. Este método busca encontrar una solución óptima o cercana a la óptima mediante un proceso iterativo que mejora continuamente la asignación. La optimización se enfoca en minimizar el tiempo total de reposición y maximizar la eficiencia en el uso de las cuadrillas de trabajo disponibles.

## Estructura del Repositorio

El repositorio contiene los siguientes archivos y directorios:

- `ahp.py`: Implementación del método AHP para priorización de activos eléctricos.
- `Asignacion.py`: Implementación del algoritmo ILS para la asignación de tareas a cuadrillas.
- `data/`: Directorio que contiene los archivos de entrada con los datos de los activos, las cuadrillas y los criterios de evaluación.
- `outputs/`: Directorio para almacenar los resultados generados por los modelos.
- `README.md`: Este archivo, que describe el propósito y la estructura del repositorio.


