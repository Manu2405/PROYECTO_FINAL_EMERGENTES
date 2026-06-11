# 🧠 TI26 - Proyecto Final de Inteligencia Artificial

## Predicción de Riesgo de Accidente Cerebrovascular mediante Redes Neuronales

### 👥 Integrantes

| Apellidos        | Nombres        |
| ---------------- | -------------- |
| Ovando Crespo    | Manuel Augusto |
| Salas Justiniano | Marco Michel   |

---

## 📖 Descripción del Proyecto

Las enfermedades cerebrovasculares constituyen una de las principales causas de mortalidad y discapacidad a nivel mundial. La detección temprana de factores de riesgo permite mejorar significativamente las estrategias de prevención y tratamiento.

El presente proyecto tiene como objetivo desarrollar un modelo de Inteligencia Artificial basado en Redes Neuronales Artificiales, específicamente un Perceptrón Multicapa (MLP), capaz de predecir la probabilidad de que un paciente sufra un Accidente Cerebrovascular (ACV o Stroke) utilizando información médica, demográfica y hábitos de vida.

Además, se realizará una comparación del desempeño de la red neuronal con otros algoritmos de Machine Learning para evaluar la efectividad de la solución propuesta.

---

## 🎯 Objetivos

### Objetivo General

Desarrollar un modelo predictivo basado en Redes Neuronales Artificiales que permita identificar pacientes con riesgo de sufrir un accidente cerebrovascular a partir de variables clínicas y demográficas.

### Objetivos Específicos

* Realizar un análisis exploratorio de datos (EDA).
* Identificar patrones y relaciones entre las variables.
* Aplicar técnicas de limpieza y transformación de datos.
* Tratar valores faltantes y datos inconsistentes.
* Codificar variables categóricas mediante técnicas de Encoding.
* Estandarizar variables numéricas para optimizar el entrenamiento.
* Balancear las clases utilizando técnicas de sobremuestreo.
* Diseñar y entrenar una Red Neuronal Multicapa (MLP).
* Evaluar el modelo mediante métricas de clasificación.
* Comparar los resultados obtenidos con modelos alternativos.

---

## 📊 Dataset Utilizado

### Stroke Prediction Dataset

**Fuente:** Kaggle

**Autor:** Fedesoriano

**Enlace:**
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

### Información General

| Característica    | Valor                 |
| ----------------- | --------------------- |
| Registros         | 5.110                 |
| Variables         | 12                    |
| Tipo de problema  | Clasificación Binaria |
| Variable Objetivo | stroke                |

---

## 📋 Descripción de Variables

| Variable          | Tipo       | Descripción               |
| ----------------- | ---------- | ------------------------- |
| id                | Numérica   | Identificador único       |
| gender            | Categórica | Género del paciente       |
| age               | Numérica   | Edad                      |
| hypertension      | Binaria    | Hipertensión              |
| heart_disease     | Binaria    | Enfermedad cardíaca       |
| ever_married      | Categórica | Estado civil              |
| work_type         | Categórica | Tipo de trabajo           |
| Residence_type    | Categórica | Tipo de residencia        |
| avg_glucose_level | Numérica   | Nivel promedio de glucosa |
| bmi               | Numérica   | Índice de masa corporal   |
| smoking_status    | Categórica | Estado de fumador         |
| stroke            | Binaria    | Variable objetivo         |

---

## 🎯 Variable Objetivo

### stroke

| Valor | Interpretación             |
| ----- | -------------------------- |
| 0     | Paciente sin riesgo de ACV |
| 1     | Paciente con riesgo de ACV |

---

## 🔄 Metodología de Desarrollo

### 1. Análisis Exploratorio de Datos (EDA)

* Análisis estadístico descriptivo.
* Distribución de variables.
* Detección de valores faltantes.
* Identificación de correlaciones.
* Visualización de patrones relevantes.

### 2. Preprocesamiento

#### Limpieza de Datos

* Eliminación de registros duplicados.
* Tratamiento de valores faltantes.
* Corrección de inconsistencias.

#### Transformación de Variables

* One-Hot Encoding para variables categóricas.
* Escalado mediante StandardScaler.

#### Balanceo de Clases

Debido al fuerte desbalance existente entre pacientes con y sin accidente cerebrovascular, se aplicará:

* SMOTE (Synthetic Minority Oversampling Technique)

#### División de Datos

* 80% Entrenamiento
* 20% Prueba

---

## 🧠 Modelo Principal

### Red Neuronal Artificial (MLP)

Arquitectura propuesta:

* Capa de Entrada
* Capa Oculta 1 (64 neuronas)
* Dropout
* Capa Oculta 2 (32 neuronas)
* Dropout
* Capa de Salida (Sigmoid)

Funciones utilizadas:

* ReLU
* Sigmoid

Optimizador:

* Adam

Función de pérdida:

* Binary Crossentropy

---

## 📈 Métricas de Evaluación

Para evaluar el desempeño del modelo se utilizarán:

* Accuracy
* Precision
* Recall
* F1-Score
* Matriz de Confusión
* Curva ROC
* Área Bajo la Curva (AUC)

---

## 🔬 Modelo Comparativo

Con el propósito de validar el desempeño de la red neuronal se implementará un modelo alternativo:

* Random Forest

Opcionalmente:

* XGBoost

Los resultados serán comparados utilizando las mismas métricas de evaluación.

---

## 📂 Estructura del Proyecto

```text
repositorio-TI26-OvandoSALAS/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── stroke_data.csv
│   │
│   └── processed/
│       └── stroke_data_clean.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_main.ipynb
│   └── 04_model_comparison.ipynb
│
└── docs/
    ├── informe_final.pdf
    └── presentacion.pdf
```

---

## 🛠️ Tecnologías Utilizadas

### Lenguaje

* Python 3.12+

### Librerías Principales

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* TensorFlow
* Keras
* XGBoost
* Imbalanced-Learn
* Missingno
* Jupyter Notebook

---

## 📅 Estado del Proyecto

| Actividad                | Estado |
| ------------------------ | ------ |
| Descarga del Dataset     | ⏳      |
| EDA                      | ⏳      |
| Preprocesamiento         | ⏳      |
| Entrenamiento MLP        | ⏳      |
| Modelo Comparativo       | ⏳      |
| Evaluación de Resultados | ⏳      |
| Informe Final            | ⏳      |
| Presentación Final       | ⏳      |

---

## 👨‍🏫 Docente

**MSc. Efraín F. Luna**

Correo:
[efrainf.luna@gmail.com](mailto:elunam@univalle.edu)

---

## 📚 Referencias

Fedesoriano. (2021). *Stroke Prediction Dataset*. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition). O'Reilly Media.

Pedregosa, F., et al. (2011). *Scikit-Learn: Machine Learning in Python*. Journal of Machine Learning Research.
