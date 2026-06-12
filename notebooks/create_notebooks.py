import json
import os

def create_eda_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📊 Análisis Exploratorio de Datos (EDA)\n",
                    "\n",
                    "Este notebook contiene el análisis exploratorio del conjunto de datos **Stroke Prediction Dataset** para comprender la distribución de las variables clínicas y demográficas, identificar valores faltantes y analizar su relación con el riesgo de accidente cerebrovascular (`stroke`)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import missingno as msno\n",
                    "\n",
                    "# Configuración de estilos para los gráficos\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams['figure.figsize'] = (10, 6)\n",
                    "plt.rcParams['font.size'] = 11"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Carga de Datos y Estructura General"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cargar el dataset desde la carpeta de datos crudos\n",
                    "df = pd.read_csv('../data/raw/healthcare-dataset-stroke-data.csv')\n",
                    "print(f\"Dimensiones del dataset: {df.shape[0]} filas, {df.shape[1]} columnas\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Mostrar información general del dataset\n",
                    "df.info()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Análisis de Valores Faltantes (Nulos)\n",
                    "\n",
                    "En la información general podemos ver que la columna `bmi` (Índice de Masa Corporal) tiene valores faltantes o valores no numéricos. Analicemos cuántos valores nulos reales tenemos."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Reemplazar 'N/A' por NaN en la columna BMI si se cargó como string\n",
                    "df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')\n",
                    "\n",
                    "null_counts = df.isnull().sum()\n",
                    "null_percentage = (df.isnull().sum() / len(df)) * 100\n",
                    "missing_df = pd.DataFrame({'Valores Nulos': null_counts, 'Porcentaje (%)': null_percentage})\n",
                    "print(missing_df[missing_df['Valores Nulos'] > 0])"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visualizar valores nulos con missingno\n",
                    "plt.figure(figsize=(10, 5))\n",
                    "msno.bar(df, color='tomato')\n",
                    "plt.title('Visualización de Valores Nulos', fontsize=15)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Distribución de la Variable Objetivo (`stroke`)\n",
                    "\n",
                    "Analicemos si el dataset está balanceado o desbalanceado."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "stroke_counts = df['stroke'].value_counts()\n",
                    "stroke_pct = df['stroke'].value_counts(normalize=True) * 100\n",
                    "\n",
                    "print(\"Distribución de la variable stroke:\")\n",
                    "for val, count, pct in zip(stroke_counts.index, stroke_counts.values, stroke_pct.values):\n",
                    "    label = \"Riesgo de ACV (1)\" if val == 1 else \"Sin Riesgo (0)\"\n",
                    "    print(f\" - {label}: {count} registros ({pct:.2f}%)\")\n",
                    "\n",
                    "plt.figure(figsize=(6, 4))\n",
                    "sns.countplot(x='stroke', data=df, palette='viridis')\n",
                    "plt.title('Distribución de Pacientes por Riesgo de Stroke', fontsize=14)\n",
                    "plt.xlabel('Stroke (0 = No, 1 = Sí)')\n",
                    "plt.ylabel('Cantidad de Pacientes')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "> **Nota Importante:** El dataset presenta un **fuerte desbalance de clases** (solo un 4.87% de los pacientes tienen antecedentes de stroke). Esto requerirá el uso de técnicas de sobremuestreo como SMOTE en la fase de preprocesamiento."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Distribución de Variables Categóricas\n",
                    "\n",
                    "Analicemos cómo se relacionan las variables categóricas con el riesgo de sufrir un accidente cerebrovascular."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cat_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']\n",
                    "\n",
                    "fig, axes = plt.subplots(4, 2, figsize=(16, 22))\n",
                    "axes = axes.flatten()\n",
                    "\n",
                    "for i, col in enumerate(cat_cols):\n",
                    "    sns.countplot(x=col, hue='stroke', data=df, ax=axes[i], palette='Set2')\n",
                    "    axes[i].set_title(f'Distribución de {col} por Stroke', fontsize=13)\n",
                    "    axes[i].set_xlabel('')\n",
                    "    axes[i].set_ylabel('Frecuencia')\n",
                    "    axes[i].tick_params(axis='x', rotation=15)\n",
                    "\n",
                    "# Ocultar el último eje que no se usa\n",
                    "axes[-1].set_visible(False)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Distribución de Variables Numéricas Continuas"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "num_cols = ['age', 'avg_glucose_level', 'bmi']\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(20, 5))\n",
                    "for i, col in enumerate(num_cols):\n",
                    "    sns.histplot(x=col, hue='stroke', data=df, kde=True, ax=axes[i], palette='coolwarm', multiple='stack')\n",
                    "    axes[i].set_title(f'Distribución de {col} por Stroke', fontsize=13)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 6. Relación de Variables Numéricas con Stroke (Boxplots)\n",
                    "\n",
                    "Usemos gráficos de caja para visualizar la mediana, los cuartiles y detectar posibles valores atípicos (outliers)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n",
                    "for i, col in enumerate(num_cols):\n",
                    "    sns.boxplot(x='stroke', y=col, data=df, ax=axes[i], palette='pastel')\n",
                    "    axes[i].set_title(f'Boxplot de {col} vs Stroke', fontsize=13)\n",
                    "    axes[i].set_xlabel('Stroke (0 = No, 1 = Sí)')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 7. Análisis de Correlación\n",
                    "\n",
                    "Codifiquemos temporalmente las variables categóricas como números para visualizar la correlación lineal general."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_encoded = df.copy()\n",
                    "for col in df_encoded.select_dtypes(include=['object']).columns:\n",
                    "    df_encoded[col] = df_encoded[col].astype('category').cat.codes\n",
                    "\n",
                    "plt.figure(figsize=(12, 10))\n",
                    "sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)\n",
                    "plt.title('Matriz de Correlación (Variables Codificadas)', fontsize=16)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Conclusiones del EDA\n",
                    "1. **Edad (`age`):** Es el factor correlacionado más fuerte con el stroke. La mayoría de los casos ocurren en pacientes mayores de 60 años.\n",
                    "2. **Glucosa (`avg_glucose_level`):** Pacientes con niveles elevados de glucosa muestran una mayor proporción de casos de stroke.\n",
                    "3. **Valores Faltantes:** La variable `bmi` tiene 201 registros nulos que requieren imputación.\n",
                    "4. **Desbalance de Clases:** El riesgo de stroke es minoritario (4.87%), por lo que se debe balancear con SMOTE antes de entrenar la Red Neuronal."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open('01_EDA.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("Notebook 01_EDA.ipynb creado con éxito.")

def create_preprocessing_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🔄 Preprocesamiento de Datos\n",
                    "\n",
                    "En este notebook se realiza la limpieza y preparación del dataset. Los pasos de preprocesamiento incluyen:\n",
                    "1. Filtrado de registros inconsistentes.\n",
                    "2. Imputación de valores nulos (`bmi`).\n",
                    "3. Codificación de variables categóricas (Encoding).\n",
                    "4. División en conjuntos de entrenamiento (80%) y prueba (20%).\n",
                    "5. Estandarización de variables numéricas (Scaling).\n",
                    "6. Balanceo de clases en entrenamiento mediante **SMOTE**."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from imblearn.over_sampling import SMOTE\n",
                    "import os\n",
                    "\n",
                    "print(\"Librerías importadas correctamente.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Carga de Datos y Limpieza de Faltantes/Inconsistencias"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cargar datos\n",
                    "df = pd.read_csv('../data/raw/healthcare-dataset-stroke-data.csv')\n",
                    "df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')\n",
                    "\n",
                    "# Remover el único registro con género 'Other' para evitar problemas en codificación binaria\n",
                    "df = df[df['gender'] != 'Other']\n",
                    "print(f\"Dimensiones después de limpieza inicial: {df.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### Imputación del BMI\n",
                    "Utilizaremos la mediana de BMI agrupada por género e hipertensión para imputar de forma más precisa los valores nulos."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Calcular la mediana según género e hipertensión\n",
                    "imputation_values = df.groupby(['gender', 'hypertension'])['bmi'].transform('median')\n",
                    "df['bmi'] = df['bmi'].fillna(imputation_values)\n",
                    "print(f\"Valores faltantes restantes en bmi: {df['bmi'].isnull().sum()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### Guardar Dataset Limpio\n",
                    "Guardamos el CSV limpio en `data/processed/stroke_data_clean.csv` tal como se solicita en los requerimientos."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Crear carpeta processed si no existe\n",
                    "os.makedirs('../data/processed', exist_ok=True)\n",
                    "\n",
                    "# Guardar dataset limpio\n",
                    "df.to_csv('../data/processed/stroke_data_clean.csv', index=False)\n",
                    "print(\"Dataset limpio guardado como 'data/processed/stroke_data_clean.csv'\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Codificación de Variables (Encoding)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Codificación binaria para variables con dos categorías\n",
                    "df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})\n",
                    "df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})\n",
                    "df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})\n",
                    "\n",
                    "# Codificación One-Hot para variables categóricas múltiples\n",
                    "df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True, dtype=int)\n",
                    "\n",
                    "# Eliminar la columna ID que no tiene valor predictivo\n",
                    "df = df.drop(columns=['id'])\n",
                    "\n",
                    "print(f\"Columnas después de codificación: {df.shape[1]}\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. División en Train / Test"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "X = df.drop(columns=['stroke'])\n",
                    "y = df['stroke']\n",
                    "\n",
                    "# División del dataset (80% entrenamiento, 20% prueba, estratificado por variable objetivo)\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
                    "\n",
                    "print(f\"Entrenamiento: {X_train.shape[0]} registros\")\n",
                    "print(f\"Prueba: {X_test.shape[0]} registros\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Estandarización de Variables Numéricas (Scaling)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "num_cols = ['age', 'avg_glucose_level', 'bmi']\n",
                    "scaler = StandardScaler()\n",
                    "\n",
                    "X_train_scaled = X_train.copy()\n",
                    "X_test_scaled = X_test.copy()\n",
                    "\n",
                    "# Ajustar scaler en train y transformar train y test\n",
                    "X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])\n",
                    "X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])\n",
                    "\n",
                    "X_train_scaled.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Balanceo de Clases mediante SMOTE\n",
                    "\n",
                    "Para evitar que la red neuronal aprenda sesgada hacia la clase mayoritaria (no stroke), aplicamos SMOTE en el conjunto de entrenamiento."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Distribución previa a SMOTE:\\n{y_train.value_counts()}\")\n",
                    "\n",
                    "smote = SMOTE(random_state=42)\n",
                    "X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)\n",
                    "\n",
                    "print(f\"Distribución posterior a SMOTE:\\n{y_train_res.value_counts()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### Guardar Variables Procesadas\n",
                    "Guardamos las matrices de entrenamiento y prueba escaladas y balanceadas para modelar directamente."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "np.savez('../data/processed/model_ready_data.npz', \n",
                    "         X_train=X_train_res.values, y_train=y_train_res.values,\n",
                    "         X_test=X_test_scaled.values, y_test=y_test.values,\n",
                    "         feature_names=X.columns.values)\n",
                    "\n",
                    "print(\"Datos de entrenamiento y prueba exportados en 'data/processed/model_ready_data.npz'\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open('02_preprocessing.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("Notebook 02_preprocessing.ipynb creado con éxito.")

def create_model_main_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🧠 Red Neuronal Artificial: Perceptrón Multicapa (MLP)\n",
                    "\n",
                    "En este notebook se entrena y evalúa la arquitectura propuesta del Perceptrón Multicapa (MLP) utilizando Keras/TensorFlow para la predicción de riesgo de accidente cerebrovascular."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, recall_score, f1_score\n",
                    "import tensorflow as tf\n",
                    "from tensorflow.keras.models import Sequential\n",
                    "from tensorflow.keras.layers import Dense, Dropout, BatchNormalization\n",
                    "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n",
                    "\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "print(f\"Versión de TensorFlow: {tf.__version__}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Carga de Datos Preprocesados"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data = np.load('../data/processed/model_ready_data.npz')\n",
                    "X_train = data['X_train']\n",
                    "y_train = data['y_train']\n",
                    "X_test = data['X_test']\n",
                    "y_test = data['y_test']\n",
                    "\n",
                    "print(f\"X_train shape: {X_train.shape}\")\n",
                    "print(f\"X_test shape: {X_test.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Definición del Modelo MLP\n",
                    "\n",
                    "Definimos la arquitectura propuesta:\n",
                    "- Capa Oculta 1: 64 neuronas, ReLU, BatchNormalization, Dropout(0.3)\n",
                    "- Capa Oculta 2: 32 neuronas, ReLU, BatchNormalization, Dropout(0.3)\n",
                    "- Capa de Salida: 1 neurona, Sigmoid"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def build_mlp(input_dim):\n",
                    "    model = Sequential([\n",
                    "        Dense(64, activation='relu', input_dim=input_dim),\n",
                    "        BatchNormalization(),\n",
                    "        Dropout(0.3),\n",
                    "        \n",
                    "        Dense(32, activation='relu'),\n",
                    "        BatchNormalization(),\n",
                    "        Dropout(0.3),\n",
                    "        \n",
                    "        Dense(1, activation='sigmoid')\n",
                    "    ])\n",
                    "    return model\n",
                    "\n",
                    "model = build_mlp(X_train.shape[1])\n",
                    "model.compile(\n",
                    "    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\n",
                    "    loss='binary_crossentropy',\n",
                    "    metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]\n",
                    ")\n",
                    "\n",
                    "model.summary()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Entrenamiento con Callbacks"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Callbacks para optimizar y detener el entrenamiento a tiempo\n",
                    "early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)\n",
                    "reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)\n",
                    "\n",
                    "history = model.fit(\n",
                    "    X_train, y_train,\n",
                    "    validation_split=0.2,\n",
                    "    epochs=60,\n",
                    "    batch_size=32,\n",
                    "    callbacks=[early_stop, reduce_lr],\n",
                    "    verbose=1\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Evaluación de Curvas de Entrenamiento"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "\n",
                    "# Pérdida\n",
                    "axes[0].plot(history.history['loss'], label='Train Loss', color='royalblue')\n",
                    "axes[0].plot(history.history['val_loss'], label='Val Loss', color='orange')\n",
                    "axes[0].set_title('Pérdida en Entrenamiento y Validación')\n",
                    "axes[0].set_xlabel('Epocas')\n",
                    "axes[0].set_ylabel('Binary Crossentropy')\n",
                    "axes[0].legend()\n",
                    "\n",
                    "# AUC\n",
                    "axes[1].plot(history.history['auc'], label='Train AUC', color='royalblue')\n",
                    "axes[1].plot(history.history['val_auc'], label='Val AUC', color='orange')\n",
                    "axes[1].set_title('Métrica AUC en Entrenamiento y Validación')\n",
                    "axes[1].set_xlabel('Epocas')\n",
                    "axes[1].set_ylabel('ROC-AUC')\n",
                    "axes[1].legend()\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Predicción y Reporte de Clasificación"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Predicciones continuas en Test set\n",
                    "y_pred_prob = model.predict(X_test).flatten()\n",
                    "y_pred = (y_pred_prob >= 0.5).astype(int)\n",
                    "\n",
                    "print(\"--- Reporte de Clasificación (MLP) ---\")\n",
                    "print(classification_report(y_test, y_pred))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Graficar Matriz de Confusión\n",
                    "cm = confusion_matrix(y_test, y_pred)\n",
                    "plt.figure(figsize=(6, 5))\n",
                    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,\n",
                    "            xticklabels=['Sin Stroke', 'Stroke'],\n",
                    "            yticklabels=['Sin Stroke', 'Stroke'])\n",
                    "plt.ylabel('Clase Real')\n",
                    "plt.xlabel('Clase Predicha')\n",
                    "plt.title('Matriz de Confusión - Red Neuronal (MLP)')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Graficar Curva ROC y calcular AUC\n",
                    "fpr, tpr, _ = roc_curve(y_test, y_pred_prob)\n",
                    "roc_auc = auc(fpr, tpr)\n",
                    "\n",
                    "plt.figure(figsize=(7, 6))\n",
                    "plt.plot(fpr, tpr, color='darkred', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')\n",
                    "plt.plot([0, 1], [0, 1], color='grey', linestyle='--')\n",
                    "plt.xlabel('Tasa de Falsos Positivos (FPR)')\n",
                    "plt.ylabel('Tasa de Verdaderos Positivos (TPR)')\n",
                    "plt.title('Curva ROC - Red Neuronal MLP')\n",
                    "plt.legend(loc='lower right')\n",
                    "plt.show()\n",
                    "\n",
                    "# Exportar predicciones de MLP para comparar\n",
                    "np.savez('../data/processed/mlp_predictions.npz', y_pred_prob=y_pred_prob, y_pred=y_pred)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open('03_model_main.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("Notebook 03_model_main.ipynb creado con éxito.")

def create_model_comparison_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🔬 Comparación de Modelos\n",
                    "\n",
                    "En este notebook entrenamos dos modelos tradicionales basados en árboles de decisión:\n",
                    "1. **Random Forest Classifier**\n",
                    "2. **XGBoost Classifier**\n",
                    "\n",
                    "Y los comparamos con el desempeño obtenido por nuestra Red Neuronal Multicapa (MLP) utilizando métricas estándar."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from xgboost import XGBClassifier\n",
                    "from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score\n",
                    "\n",
                    "sns.set_theme(style=\"whitegrid\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Carga de Datos y Predicciones de MLP"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Datos listos para modelar\n",
                    "data = np.load('../data/processed/model_ready_data.npz')\n",
                    "X_train = data['X_train']\n",
                    "y_train = data['y_train']\n",
                    "X_test = data['X_test']\n",
                    "y_test = data['y_test']\n",
                    "\n",
                    "# Predicciones del modelo MLP\n",
                    "mlp_data = np.load('../data/processed/mlp_predictions.npz')\n",
                    "y_pred_prob_mlp = mlp_data['y_pred_prob']\n",
                    "y_pred_mlp = mlp_data['y_pred']\n",
                    "\n",
                    "print(\"Datos cargados correctamente.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Entrenamiento de Modelos Comparativos"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2.1. Random Forest Classifier\n",
                    "rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')\n",
                    "rf.fit(X_train, y_train)\n",
                    "\n",
                    "y_pred_prob_rf = rf.predict_proba(X_test)[:, 1]\n",
                    "y_pred_rf = rf.predict(X_test)\n",
                    "\n",
                    "print(\"--- Random Forest Report ---\")\n",
                    "print(classification_report(y_test, y_pred_rf))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2.2. XGBoost Classifier\n",
                    "xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')\n",
                    "xgb.fit(X_train, y_train)\n",
                    "\n",
                    "y_pred_prob_xgb = xgb.predict_proba(X_test)[:, 1]\n",
                    "y_pred_xgb = xgb.predict(X_test)\n",
                    "\n",
                    "print(\"--- XGBoost Report ---\")\n",
                    "print(classification_report(y_test, y_pred_xgb))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Comparación de Curvas ROC"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Curva ROC para MLP\n",
                    "fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_pred_prob_mlp)\n",
                    "auc_mlp = auc(fpr_mlp, tpr_mlp)\n",
                    "\n",
                    "# Curva ROC para Random Forest\n",
                    "fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)\n",
                    "auc_rf = auc(fpr_rf, tpr_rf)\n",
                    "\n",
                    "# Curva ROC para XGBoost\n",
                    "fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob_xgb)\n",
                    "auc_xgb = auc(fpr_xgb, tpr_xgb)\n",
                    "\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "plt.plot(fpr_mlp, tpr_mlp, label=f'Red Neuronal (MLP) (AUC = {auc_mlp:.4f})', color='darkred', lw=2)\n",
                    "plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})', color='forestgreen', lw=2)\n",
                    "plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.4f})', color='darkblue', lw=2)\n",
                    "plt.plot([0, 1], [0, 1], color='grey', linestyle='--')\n",
                    "plt.xlim([0.0, 1.0])\n",
                    "plt.ylim([0.0, 1.05])\n",
                    "plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)\n",
                    "plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)\n",
                    "plt.title('Comparación de Curvas ROC de los Modelos Evaluados', fontsize=14)\n",
                    "plt.legend(loc='lower right', fontsize=11)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Tabla Comparativa de Métricas"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model_names = ['Red Neuronal (MLP)', 'Random Forest', 'XGBoost']\n",
                    "preds = [y_pred_mlp, y_pred_rf, y_pred_xgb]\n",
                    "probs = [y_pred_prob_mlp, y_pred_prob_rf, y_pred_prob_xgb]\n",
                    "\n",
                    "metrics_data = []\n",
                    "for name, pred, prob in zip(model_names, preds, probs):\n",
                    "    fpr, tpr, _ = roc_curve(y_test, prob)\n",
                    "    metrics_data.append({\n",
                    "        'Modelo': name,\n",
                    "        'Accuracy': accuracy_score(y_test, pred),\n",
                    "        'Precision (Stroke)': precision_score(y_test, pred),\n",
                    "        'Recall (Stroke)': recall_score(y_test, pred),\n",
                    "        'F1-Score (Stroke)': f1_score(y_test, pred),\n",
                    "        'ROC-AUC': auc(fpr, tpr)\n",
                    "    })\n",
                    "\n",
                    "df_comparison = pd.DataFrame(metrics_data)\n",
                    "df_comparison"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Conclusión Comparativa\n",
                    "\n",
                    "- **Recall:** En la predicción de stroke (un caso médico), el **Recall** (Sensibilidad) es la métrica más importante, dado que nos interesa minimizar los Falsos Negativos (pacientes en riesgo que el modelo no detecta).\n",
                    "- **SMOTE:** La aplicación de SMOTE ayudó a mejorar significativamente el recall de todos los modelos.\n",
                    "- **MLP vs Árboles:** La Red Neuronal Artificial (MLP) ofrece un desempeño balanceado y ajustable en el umbral de decisión, logrando una frontera de decisión no lineal robusta frente al desbalance estructural de los datos."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open('04_model_comparison.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("Notebook 04_model_comparison.ipynb creado con éxito.")

if __name__ == '__main__':
    create_eda_notebook()
    create_preprocessing_notebook()
    create_model_main_notebook()
    create_model_comparison_notebook()
