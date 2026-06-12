import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_plots():
    clean_csv_path = '../data/processed/stroke_data_clean.csv'
    plots_dir = '../data/processed/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Cargando el dataset limpio para generar gráficos...")
    df = pd.read_csv(clean_csv_path)
    
    # Configuración general de estilo visual (premium y limpio)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # --- 1. Distribución de la variable objetivo (Stroke) ---
    print("1. Generando gráfico de distribución de stroke...")
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df['stroke'].value_counts()
    colors = ['#4A90E2', '#D0021B'] # Azul premium y rojo soft
    
    bars = ax.bar(['Sin Stroke (0)', 'Con Stroke (1)'], counts, color=colors, width=0.6, edgecolor='grey', alpha=0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Agregar etiquetas sobre las barras
    for bar in bars:
        height = bar.get_height()
        percentage = (height / len(df)) * 100
        ax.annotate(f'{height}\n({percentage:.2f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    ax.set_title('Distribución de Pacientes por Riesgo de Stroke', fontsize=14, pad=15, fontweight='bold', color='#333333')
    ax.set_ylabel('Cantidad de Pacientes', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, '01_distribution_stroke.png'), dpi=150)
    plt.close()
    
    # --- 2. Distribución de edad por Stroke (Histograma) ---
    print("2. Generando gráfico de distribución de edad por stroke...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Separar los datos
    age_no_stroke = df[df['stroke'] == 0]['age']
    age_stroke = df[df['stroke'] == 1]['age']
    
    ax.hist(age_no_stroke, bins=25, alpha=0.6, label='Sin Stroke', color='#4A90E2', edgecolor='white')
    ax.hist(age_stroke, bins=25, alpha=0.8, label='Con Stroke', color='#D0021B', edgecolor='white')
    
    ax.set_title('Distribución de Edad de los Pacientes por Stroke', fontsize=14, pad=15, fontweight='bold', color='#333333')
    ax.set_xlabel('Edad (Años)', fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    ax.legend(frameon=True, facecolor='white', edgecolor='#CCCCCC')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, '02_age_distribution.png'), dpi=150)
    plt.close()
    
    # --- 3. Boxplots de variables numéricas continuas ---
    print("3. Generando boxplots de variables numéricas...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    titles = ['Edad', 'Nivel Promedio de Glucosa', 'Índice de Masa Corporal (BMI)']
    
    for i, col in enumerate(num_cols):
        data_to_plot = [df[df['stroke'] == 0][col], df[df['stroke'] == 1][col]]
        box = axes[i].boxplot(data_to_plot, patch_artist=True, tick_labels=['Sin Stroke', 'Con Stroke'], widths=0.5)
        
        # Estilos para las cajas
        colors = ['#E6F0FA', '#FFE6E6']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('grey')
        
        for median in box['medians']:
            median.set(color='black', linewidth=1.5)
            
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.suptitle('Comparativa de Variables Numéricas Continuas vs Stroke', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, '03_numerical_boxplots.png'), dpi=150)
    plt.close()
    
    # --- 4. Correlación Heatmap ---
    print("4. Generando heatmap de correlación...")
    # Codificar variables categóricas para la matriz de correlación
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
        
    corr = df_encoded.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # Dibujar un mapa de calor a mano (o con matplotlib matshow)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    
    # Configurar etiquetas
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    
    # Mostrar coeficientes en cada celda
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, f'{z:.2f}', ha='center', va='center',
                color='black' if abs(z) < 0.6 else 'white', fontsize=8)
                
    ax.set_title('Matriz de Correlación de Variables', fontsize=14, pad=20, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, '04_correlation_heatmap.png'), dpi=150)
    plt.close()
    
    print(f"¡Todos los gráficos fueron generados con éxito en la carpeta: {plots_dir}!")

if __name__ == '__main__':
    # Cambiar de directorio para asegurar rutas relativas correctas
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generate_plots()
