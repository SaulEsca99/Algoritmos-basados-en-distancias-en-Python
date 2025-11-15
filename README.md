# 📊 Actividad 1: Algoritmos basados en distancias en Python  
## MACHINE LEARNING

**Integrantes:**
* González Pérez Monserrat  
* Escamilla Lazcano Saúl  
* Pérez Méndez Nancy Esmeralda  
* Valencia Hernandez Kevin Guadalupe  
* Zamudio López Leonardo  

**Grupo:** 5BV1  
**Profesora:** Dra. Camacho Vázquez Vanessa Alejandra  
**Fecha:** 03/09/25  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Arrays-orange.svg)](https://numpy.org/)

---

## 🚀 Descripción general del proyecto

Este repositorio contiene la implementación en Python de algoritmos de clasificación **basados en distancias**, principalmente:

- **K-Nearest Neighbors (K-NN)**
- **Clasificador de Mínima Distancia (basado en centroides)**

Además, se incluyen:

- Utilidades para **carga y preprocesamiento básico de datos** desde archivos de texto/CSV.
- Implementación de **métodos de validación** para evaluar el desempeño de los clasificadores:
  - Train/Test
  - K-Fold Cross Validation
  - Bootstrap

El código está orientado a ser **interactivo en consola**, permitiendo al usuario:
1. Cargar diferentes archivos de datos.
2. Seleccionar qué columnas serán **features (entrada)** y cuál será la **clase (salida)**.
3. Probar clasificadores con puntos ingresados manualmente.
4. Evaluar el rendimiento de los algoritmos con distintos métodos de validación.

---

## 🗂️ Estructura del repositorio

- `AlgoritmosBasadosEnDistancias.py`  
  Programa principal para:
  - Cargar datos desde CSV.
  - Seleccionar columnas de entrada/salida.
  - Calcular centroides.
  - Clasificar puntos nuevos con:
    - Mínima Distancia (centroides)
    - K-NN (con K configurable)
  - Mostrar estadísticas descriptivas básicas.

- `MetodosValidacion.py`  
  Implementa los métodos de validación:
  - **Train/Test**
  - **K-Fold Cross Validation**
  - **Bootstrap**
  
  Todo utilizando como clasificadores:
  - Mínima Distancia
  - K-NN

- `CargaDatos.py`  
  Módulo general para:
  - Cargar archivos de texto separados por un delimitador.
  - Determinar el **tipo de dato** de cada columna (entero, flotante, booleano, cadena).
  - Determinar el **tipo de medida** (discreta, continua, nominal).
  - Seleccionar subconjuntos de **atributos** (columnas) y **renglones** (filas).
  - Guardar matrices reducidas en nuevos archivos.

- Conjuntos de datos de ejemplo (formato texto/CSV):
  - `iris.data`
  - `zoo.data`
  - `wine.data`

---

## 🧮 1. Algoritmos basados en distancias

El núcleo del proyecto es la implementación de clasificadores que deciden la clase de un patrón nuevo a partir de **distancias** en el espacio de características.

### 🔹 Distancias implementadas

En `AlgoritmosBasadosEnDistancias.py` se utilizan principalmente:

- **Distancia euclidiana**  
  \[
  d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i}(x_i - y_i)^2}
  \]

- **Distancia Manhattan** (solo en el módulo interactivo de clasificación)  
  \[
  d(\mathbf{x}, \mathbf{y}) = \sum_{i}|x_i - y_i|
  \]

Estas funciones reciben dos vectores numéricos (lista o arreglo de NumPy) y devuelven la distancia entre ellos.

---

## 🧱 2. Clasificador de Mínima Distancia (Centroides)

En este método:

1. Para cada clase se calcula su **centroide**:
   - Es el **promedio** de todos los vectores de entrenamiento pertenecientes a esa clase.
2. Para un nuevo punto a clasificar:
   - Se calcula la distancia a cada centroide.
   - Se asigna la clase del centroide **más cercano**.

El módulo:
- Calcula y muestra para cada clase:
  - Número de patrones.
  - Centroide correspondiente.
- Permite elegir la métrica de distancia:
  - Euclidiana.
  - Manhattan (en modo interactivo).

---

## 🔄 3. Clasificador K-Nearest Neighbors (K-NN)

K-NN es un clasificador **no paramétrico** que funciona así:

1. Dado un punto nuevo, calcula la distancia a **todos** los puntos de entrenamiento.
2. Ordena las distancias de menor a mayor.
3. Toma los **K vecinos más cercanos**.
4. La clase predicha es la **más frecuente** entre esos K vecinos (votación mayoritaria).

En el código se permite:
- Configurar el valor de **K** (si no es válido, se usa K = 3 por defecto).
- Elegir el tipo de distancia (euclidiana o manhattan, según el módulo).

El programa también imprime, a modo de depuración:
- Algunas de las primeras distancias calculadas.
- Los K vecinos más cercanos:
  - Índice en el conjunto de entrenamiento.
  - Clase.
  - Distancia.

---

## 📊 4. Métodos de validación implementados

En `MetodosValidacion.py` se incluyen varias estrategias para estimar el rendimiento de los clasificadores:

### 4.1 Train/Test

- Se divide el conjunto de datos en:
  - **Entrenamiento**: porcentaje configurable (ej. 70%).
  - **Prueba**: el resto (ej. 30%).
- Se entrena el clasificador con el conjunto de entrenamiento.
- Se prueba en el conjunto de prueba.
- Se calculan:
  - Porcentaje de **eficiencia (accuracy)**.
  - Porcentaje de **error**.
  - Número de patrones correctamente clasificados.

### 4.2 K-Fold Cross Validation

- Se elige un número **K** de folds (por defecto 5).
- Se barajan los datos y se repite:
  - Para cada fold:
    - Ese fold se usa como **prueba**.
    - Los demás folds se usan como **entrenamiento**.
- Se promedian:
  - Eficiencia.
  - Error.
  - Se calcula también la **desviación estándar** de ambas métricas.

### 4.3 Bootstrap

- Se repiten **K experimentos** (configurable, por defecto 10).
- En cada experimento:
  - Conjunto de aprendizaje: muestreo **con reemplazo**.
  - Conjunto de prueba: muestreo **sin reemplazo**.
- Se calcula:
  - Eficiencia y error globales por experimento.
  - Eficiencia y error **por clase**, con promedio y desviación estándar.

---

## 📥 5. Carga y análisis básico de datos

### 5.1 Desde CSV (Pandas)

Tanto `AlgoritmosBasadosEnDistancias.py` como `MetodosValidacion.py`:

- Preguntan si el archivo tiene **encabezados** (nombres de columnas).
- Cargan el archivo con `pandas.read_csv`.
- Muestran:
  - Dimensiones del dataset.
  - Nombres de columnas.
  - Primeras filas.
- Permiten elegir por teclado:
  - Columnas de **entrada (features)**.
  - Columna de **salida (clase)**.

### 5.2 Desde archivo de texto genérico

`CargaDatos.py`:

- Pide:
  - Ruta del archivo.
  - Delimitador (por ejemplo `,` o `;`).
- Construye una **matriz** (lista de listas) con todos los valores.
- Determina:
  - Tipo de dato predominante por columna (entero, flotante, booleano, cadena).
  - Tipo de medida (discreta, continua, nominal).
- Permite:
  - Seleccionar subconjuntos de **atributos**.
  - Seleccionar subconjuntos de **renglones**.
  - Guardar matrices reducidas en nuevos archivos.

---

## 🧪 6. Conjuntos de datos de ejemplo

En el repositorio se incluyen archivos clásicos de aprendizaje de máquina (formato texto/CSV):

- `iris.data` – Conjunto de datos de flores Iris.
- `zoo.data` – Conjunto de datos de animales con atributos booleanos/categóricos.
- `wine.data` – Conjunto de datos de vinos con atributos numéricos.

Estos archivos se pueden usar tanto para:
- Probar los **clasificadores basados en distancias**.
- Probar los **métodos de validación**.

---

## 🛠️ 7. Requisitos

- **Python** 3.10.19
- Librerías de Python:
  - `pandas`
  - `numpy`
  - `math` (estándar)
  - `collections` (estándar)
  - `random` (estándar)

