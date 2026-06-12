import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Preprocesamiento de datos ---
def load_and_preprocess():
    clean_csv_path = '../data/processed/stroke_data_clean.csv'
    df = pd.read_csv(clean_csv_path)
    
    # Codificación manual de variables categóricas
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
    df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
    
    # One-hot encoding manual para variables con múltiples categorías
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True, dtype=int)
    
    # Eliminar columna ID
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    X = df.drop(columns=['stroke']).values.astype(float)
    y = df['stroke'].values.astype(float).reshape(-1, 1)
    
    return X, y, df.drop(columns=['stroke']).columns

# --- 2. División Entrenamiento/Prueba y Estandarización ---
def train_test_split_manual(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Normalización (Estandarización)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0 # Evitar división por cero
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# --- 3. Balanceo de clases (Random Oversampling) ---
def oversample_class(X, y):
    y_flat = y.flatten()
    X_0 = X[y_flat == 0]
    X_1 = X[y_flat == 1]
    
    count_0 = X_0.shape[0]
    count_1 = X_1.shape[0]
    
    if count_0 > count_1:
        # Sobremuestrear clase 1 para que tenga la misma cantidad que clase 0
        idx_resampled = np.random.choice(count_1, size=count_0, replace=True)
        X_1_res = X_1[idx_resampled]
        y_1_res = np.ones((count_0, 1))
        
        X_res = np.vstack([X_0, X_1_res])
        y_res = np.vstack([np.zeros((count_0, 1)), y_1_res])
    else:
        # Sobremuestrear clase 0
        idx_resampled = np.random.choice(count_0, size=count_1, replace=True)
        X_0_res = X_0[idx_resampled]
        y_0_res = np.zeros((count_1, 1))
        
        X_res = np.vstack([X_0_res, X_1])
        y_res = np.vstack([y_0_res, np.ones((count_1, 1))])
        
    # Mezclar datos finales
    mix_idx = np.arange(X_res.shape[0])
    np.random.shuffle(mix_idx)
    return X_res[mix_idx], y_res[mix_idx]

# --- 4. Red Neuronal Artificial (MLP) en NumPy ---
class MLPFromScratch:
    def __init__(self, input_dim, hidden1=64, hidden2=32, lr=0.001, seed=42):
        np.random.seed(seed)
        self.lr = lr
        
        # Inicialización de pesos de He (Kaiming) para ReLU
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden1))
        
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        
        # Inicialización de Xavier para Sigmoid
        self.W3 = np.random.randn(hidden2, 1) * np.sqrt(1.0 / hidden2)
        self.b3 = np.zeros((1, 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))

    def forward(self, X):
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)
        return self.A3

    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        # Agregar pequeña epsilon para evitar log(0)
        eps = 1e-15
        loss = - (1.0 / m) * np.sum(y * np.log(y_hat + eps) + (1.0 - y) * np.log(1.0 - y_hat + eps))
        return loss

    def backward(self, y, y_hat):
        m = y.shape[0]
        
        # Derivadas
        dZ3 = y_hat - y
        self.dW3 = np.dot(self.A2.T, dZ3) / m
        self.db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (self.Z2 > 0) # ReLU derivative
        self.dW2 = np.dot(self.A1.T, dZ2) / m
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0) # ReLU derivative
        self.dW1 = np.dot(self.X.T, dZ1) / m
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    def update_parameters(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2
        
        self.W3 -= self.lr * self.dW3
        self.b3 -= self.lr * self.db3

# --- 5. Métricas de Evaluación ---
def evaluate_metrics(y_true, y_pred_prob):
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Curva ROC
    thresholds = np.linspace(0, 1, 150)
    tpr_list = []
    fpr_list = []
    
    for t in thresholds:
        yp = (y_pred_prob >= t).astype(int)
        tp_t = np.sum((y_true == 1) & (yp == 1))
        tn_t = np.sum((y_true == 0) & (yp == 0))
        fp_t = np.sum((y_true == 0) & (yp == 1))
        fn_t = np.sum((y_true == 1) & (yp == 0))
        
        tpr_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
        
        tpr_list.append(tpr_t)
        fpr_list.append(fpr_t)
        
    # Ordenar FPR y TPR para calcular AUC (trapz)
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    
    # Invertir el orden ya que thresholds van de 0 a 1 (por lo tanto FPR va de 1 a 0)
    fpr_arr = fpr_arr[::-1]
    tpr_arr = tpr_arr[::-1]
    
    auc_val = np.sum((fpr_arr[1:] - fpr_arr[:-1]) * (tpr_arr[1:] + tpr_arr[:-1]) / 2.0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'fpr': fpr_arr, 'tpr': tpr_arr, 'auc': auc_val
    }

# --- 6. Función Principal ---
def run_pipeline():
    X, y, feature_names = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
    
    # Guardar distribución de clases previa
    train_counts = np.bincount(y_train.flatten().astype(int))
    print(f"Distribución original en train: 0={train_counts[0]}, 1={train_counts[1]}")
    
    # Balancear clases
    X_train_res, y_train_res = oversample_class(X_train, y_train)
    res_counts = np.bincount(y_train_res.flatten().astype(int))
    print(f"Distribución balanceada en train: 0={res_counts[0]}, 1={res_counts[1]}")
    
    # Crear y entrenar red neuronal
    mlp = MLPFromScratch(input_dim=X_train.shape[1], hidden1=64, hidden2=32, lr=0.01)
    
    losses = []
    epochs = 1200
    
    print("\nIniciando entrenamiento de la red neuronal (1200 épocas)...")
    for epoch in range(epochs):
        y_hat = mlp.forward(X_train_res)
        loss = mlp.compute_loss(y_train_res, y_hat)
        losses.append(loss)
        
        mlp.backward(y_train_res, y_hat)
        mlp.update_parameters()
        
        if (epoch + 1) % 100 == 0:
            print(f"Época {epoch+1}/{epochs} - Loss: {loss:.5f}")
            
    # Guardar Curva de Pérdida
    plots_dir = '../data/processed/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(7, 4))
    plt.plot(losses, color='darkred', label='Crossentropy Loss')
    plt.title('Curva de Pérdida durante el Entrenamiento del MLP', fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '05_mlp_loss_curve.png'), dpi=150)
    plt.close()
    
    # Predicción y Evaluación
    y_pred_prob = mlp.forward(X_test)
    metrics = evaluate_metrics(y_test, y_pred_prob)
    
    print("\n--- Resultados del Perceptrón Multicapa (MLP) ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['auc']:.4f}")
    print(f"Matriz de Confusión: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")
    
    # Guardar Matriz de Confusión como gráfico
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    
    # Dibujar heatmap a mano con matplotlib
    cax = ax.matshow(cm, cmap='Blues', alpha=0.7)
    fig.colorbar(cax)
    
    ax.set_xticklabels(['', 'Sin Stroke', 'Stroke'])
    ax.set_yticklabels(['', 'Sin Stroke', 'Stroke'])
    ax.set_xlabel('Predicción del Modelo', fontweight='bold', labelpad=10)
    ax.set_ylabel('Clase Real', fontweight='bold', labelpad=10)
    
    # Rellenar valores
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f'{z}', ha='center', va='center', fontsize=12, fontweight='bold')
        
    ax.set_title('Matriz de Confusión (MLP en NumPy)', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '06_mlp_confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Guardar Curva ROC
    plt.figure(figsize=(6, 5))
    plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, label=f'ROC Curve (AUC = {metrics['auc']:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - MLP en NumPy', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '07_mlp_roc_curve.png'), dpi=150)
    plt.close()
    
    print("\n¡Resultados de la Red Neuronal y gráficos generados con éxito!")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_pipeline()
