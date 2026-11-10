# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:39:47 2026

@author: ADMIN
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df=pd.read_csv("C:/Users/ADMIN/Documents/PVK/Crystal_structure.csv")

# 1. Definimos explícitamente la lista de características numéricas físicas (excluyendo texto)
features = [
    "v(A)",
    "v(B)",
    "r(AXII)(Å)",
    "r(AVI)(Å)",
    "r(BVI)(Å)",
    "EN(A)",
    "EN(B)",
    "l(A-O)(Å)",
    "l(B-O)(Å)",
    "ΔENR",
    "tG",
    "μ",
]
# como ejercicio verificar si es necesaria el uso de replace cuando ya se utiliza error=coerce

df["v(A)"] = df["v(A)"].replace('-', np.nan)
df['v(A)'] = pd.to_numeric(df['v(A)'], errors='coerce')

df["v(B)"] = df["v(B)"].replace('-', np.nan)
df['v(B)'] = pd.to_numeric(df['v(B)'], errors='coerce')

df["τ"] = df["τ"].replace('-', np.nan)
df['τ'] = pd.to_numeric(df['τ'], errors='coerce')

features = [
    "v(A)",
    "v(B)",
    "r(AXII)(Å)",
    "r(AVI)(Å)",
    "r(BVI)(Å)",
    "EN(A)",
    "EN(B)",
    "l(A-O)(Å)",
    "l(B-O)(Å)",
    "ΔENR",
    "tG",
     "τ",
    "μ"
]
#print(df.info())

target_column = "Lowest distortion"

# CORRECCIÓN 1: Eliminar filas donde la variable objetivo sea un guión '-' o nula
df = df[df[target_column] != "-"]
df = df.dropna(subset=[target_column])


X = df[features].copy()
y = df[target_column]

# 3. Imputación controlada de nulos en las características (usando la mediana)

X = X.fillna(X.median())

# 4. División estratificada (crucial para mantener la proporción de las clases minoritarias)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Escalamiento de datos (indispensable para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Entrenamiento con mitigación de desbalance (class_weight='balanced')
print("--- Entrenando Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


print("--- Entrenando Máquina de Soporte Vectorial (SVM) ---")
svm_model = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

# 7. Evaluación basada en métricas reales (Reporte por clase y Macro F1)
print("\n==============================")
print("RESULTADOS: RANDOM FOREST")
print("==============================")
print(classification_report(y_test, y_pred_rf, zero_division=0))

print("\n==============================")
print("RESULTADOS: SVM (MSV)")
print("==============================")
print(classification_report(y_test, y_pred_svm, zero_division=0))


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 1. Calcular la matriz de confusión normalizada por filas (recall por clase)
cm = confusion_matrix(y_test, y_pred_rf, normalize="true")

# # 2. Graficar con Seaborn / Matplotlib
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt=".2f",
#     cmap="Blues",
#     xticklabels=rf_model.classes_,
#     yticklabels=rf_model.classes_,
# )
# plt.title(
#     "Matriz de Confusión Normalizada - Random Forest (Perovskitas)",
#     fontsize=12,
# )
# plt.xlabel("Predicción del Modelo", fontsize=10)
# plt.ylabel("Clase Real (Ground Truth)", fontsize=10)
# plt.tight_layout()
# plt.show()

# Extracción estricta de importancia de características
importances = rf_model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
).sort_values(by="Importance", ascending=False)

print("--- IMPORTANCIA DE CARACTERÍSTICAS (RANDOM FOREST) ---")
print(feature_importance_df)

# Gráfica para el repositorio
plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance", y="Feature", data=feature_importance_df, palette="viridis"
)
plt.title(
    "Importancia de Características - Predicción de Distorsión en Perovskitas"
)
plt.xlabel("Importancia Relativa (Gini Importance)")
plt.ylabel("Características Físicas")
plt.tight_layout()
plt.show()













