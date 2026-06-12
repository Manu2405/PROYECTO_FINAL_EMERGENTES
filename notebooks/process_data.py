import os
import pandas as pd
import numpy as np

def clean_data():
    raw_path = '../data/raw/healthcare-dataset-stroke-data.csv'
    processed_dir = '../data/processed'
    processed_path = os.path.join(processed_dir, 'stroke_data_clean.csv')

    print("Cargando el dataset original...")
    df = pd.read_csv(raw_path)
    
    # 1. Tratamiento de bmi (convertir 'N/A' a nulo)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    
    # 2. Filtrado de inconsistencia de género ('Other')
    initial_rows = df.shape[0]
    df = df[df['gender'] != 'Other']
    print(f"Filas eliminadas por género inconsistente: {initial_rows - df.shape[0]}")
    
    # 3. Imputación de valores faltantes en bmi
    # Agrupamos por género e hipertensión para calcular la mediana y tener una mejor estimación
    median_by_group = df.groupby(['gender', 'hypertension'])['bmi'].transform('median')
    df['bmi'] = df['bmi'].fillna(median_by_group)
    print(f"Valores faltantes en bmi después de la imputación: {df['bmi'].isnull().sum()}")
    
    # Asegurar que existe la carpeta processed
    os.makedirs(processed_dir, exist_ok=True)
    
    # Guardar el archivo limpio filtrado
    df.to_csv(processed_path, index=False)
    print(f"Datos correctamente procesados y guardados en: {processed_path}")
    print(f"Dimensión del CSV guardado: {df.shape}")

if __name__ == '__main__':
    # Cambiar de directorio para asegurar rutas relativas correctas si se ejecuta desde notebooks
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    clean_data()
