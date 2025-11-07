"""
MODELO DE PREDICCIÓN DE VICTORIAS EN UEFA CHAMPIONS LEAGUE
==========================================================
Versión adaptada para CSV con estadísticas reales de equipos
Variables: Equipo, Partidos, Ganados, Empates, Perdidos, Goles, Disparos, Ataques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("MODELO DE PREDICCIÓN UEFA CHAMPIONS LEAGUE - Probabilidad de Victoria")
print("="*80)

# =============================================================================
# FASE 1: CARGA DE DATOS DESDE CSV (RESTRUCTURADA)
# =============================================================================
print("\n[FASE 1] CARGA DE DATOS DESDE CSV")
print("-"*80)

# IMPORTANTE: Coloca aquí la ruta de tu archivo CSV
# Puede ser una ruta local o una URL
CSV_PATH = input("\n📂 Ingresa la ruta o URL de tu archivo CSV: ").strip()

# Si prefieres hardcodear la ruta, descomenta y modifica esta línea:
# CSV_PATH = "ruta/a/tu/archivo.csv"

print(f"\n🔄 Cargando datos desde: {CSV_PATH}")

try:
    # Intentar cargar el CSV
    if CSV_PATH.startswith('http'):
        df = pd.read_csv(CSV_PATH)
        print("✓ CSV cargado exitosamente desde URL")
    else:
        df = pd.read_csv(CSV_PATH)
        print("✓ CSV cargado exitosamente desde archivo local")
    
    print(f"\n📊 Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Mostrar primeras filas
    print(f"\n📋 Primeras 5 filas del dataset:")
    print(df.head())
    
    # Información de columnas
    print(f"\n📋 Columnas detectadas:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    print(f"\n📊 Información del dataset:")
    print(df.info())
    
except FileNotFoundError:
    print(f"\n❌ ERROR: No se encontró el archivo en la ruta especificada")
    print(f"   Verifica que la ruta sea correcta: {CSV_PATH}")
    exit()
except Exception as e:
    print(f"\n❌ ERROR al cargar el CSV: {str(e)}")
    exit()

# =============================================================================
# FASE 2: LIMPIEZA Y PREPROCESAMIENTO
# =============================================================================
print("\n" + "="*80)
print("[FASE 2] LIMPIEZA Y PREPROCESAMIENTO")
print("-"*80)

# Normalizar nombres de columnas (eliminar espacios, mayúsculas)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(f"\n✓ Nombres de columnas normalizados")

# Mostrar columnas después de normalización
print(f"\n📋 Columnas normalizadas:")
for col in df.columns:
    print(f"   • {col}")

# Verificar valores nulos
print(f"\n🔍 Verificando valores nulos:")
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print(null_counts[null_counts > 0])
    print(f"\n🧹 Eliminando filas con valores nulos...")
    df = df.dropna()
    print(f"   ✓ {null_counts.sum()} valores nulos eliminados")
else:
    print("   ✓ No se encontraron valores nulos")

# Eliminar duplicados
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"\n🧹 Eliminando {duplicates} filas duplicadas...")
    df = df.drop_duplicates()
else:
    print(f"\n✓ No se encontraron duplicados")

# Convertir columnas numéricas
numeric_columns = [col for col in df.columns if col != 'equipo']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\n✓ Dataset limpio: {df.shape[0]} equipos")

# Verificar que tenemos las columnas esperadas
expected_cols = ['equipo', 'partidos_disputados', 'ganados', 'empates', 
                'perdidos', 'goles', 'disparos_totales', 'ataques']

# Mapeo flexible de nombres de columnas
column_mapping = {}
for expected in expected_cols:
    for actual in df.columns:
        if expected.replace('_', '') in actual.replace('_', ''):
            column_mapping[actual] = expected
            break

if column_mapping:
    df = df.rename(columns=column_mapping)
    print(f"\n✓ Columnas mapeadas correctamente")

# Mostrar estadísticas básicas
print(f"\n📊 Estadísticas básicas del dataset:")
print(df.describe())

# =============================================================================
# FASE 3: INGENIERÍA DE CARACTERÍSTICAS
# =============================================================================
print("\n" + "="*80)
print("[FASE 3] INGENIERÍA DE CARACTERÍSTICAS")
print("-"*80)

# Crear características derivadas basadas en las variables disponibles
print(f"\n🔧 Creando características derivadas...\n")

# 1. Porcentaje de victorias (variable objetivo principal)
df['win_rate'] = (df['ganados'] / df['partidos_disputados']) * 100
print(f"   ✓ win_rate: Porcentaje de victorias")

# 2. Puntos obtenidos (3 por victoria, 1 por empate)
df['puntos'] = (df['ganados'] * 3) + (df['empates'] * 1)
print(f"   ✓ puntos: Total de puntos obtenidos")

# 3. Puntos por partido
df['puntos_por_partido'] = df['puntos'] / df['partidos_disputados']
print(f"   ✓ puntos_por_partido: Promedio de puntos")

# 4. Goles por partido (eficiencia ofensiva)
df['goles_por_partido'] = df['goles'] / df['partidos_disputados']
print(f"   ✓ goles_por_partido: Eficiencia ofensiva")

# 5. Disparos por partido
df['disparos_por_partido'] = df['disparos_totales'] / df['partidos_disputados']
print(f"   ✓ disparos_por_partido: Volumen ofensivo")

# 6. Ataques por partido
df['ataques_por_partido'] = df['ataques'] / df['partidos_disputados']
print(f"   ✓ ataques_por_partido: Presión ofensiva")

# 7. Eficiencia de disparo (goles / disparos)
df['eficiencia_disparo'] = (df['goles'] / df['disparos_totales']) * 100
print(f"   ✓ eficiencia_disparo: Conversión de disparos")

# 8. Eficiencia de ataque (goles / ataques)
df['eficiencia_ataque'] = (df['goles'] / df['ataques']) * 100
print(f"   ✓ eficiencia_ataque: Conversión de ataques")

# 9. Ratio disparos/ataques (calidad de ataques)
df['ratio_disparos_ataques'] = (df['disparos_totales'] / df['ataques']) * 100
print(f"   ✓ ratio_disparos_ataques: Calidad de ataques")

# 10. Porcentaje de derrotas (riesgo)
df['defeat_rate'] = (df['perdidos'] / df['partidos_disputados']) * 100
print(f"   ✓ defeat_rate: Porcentaje de derrotas")

# 11. Porcentaje de empates (consistencia)
df['draw_rate'] = (df['empates'] / df['partidos_disputados']) * 100
print(f"   ✓ draw_rate: Porcentaje de empates")

# 12. Índice de rendimiento compuesto
df['performance_index'] = (
    df['win_rate'] * 0.4 +
    df['goles_por_partido'] * 10 +
    df['eficiencia_disparo'] * 0.3 +
    df['puntos_por_partido'] * 10
)
print(f"   ✓ performance_index: Índice compuesto de rendimiento")

# Reemplazar infinitos y NaN resultantes de divisiones por cero
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

print(f"\n✅ Total de características: {len(df.columns) - 1} (excluyendo 'equipo')")

# =============================================================================
# FASE 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
print("\n" + "="*80)
print("[FASE 4] ANÁLISIS EXPLORATORIO DE DATOS")
print("-"*80)

# Ranking de equipos por win_rate
df_sorted = df.sort_values('win_rate', ascending=False)

print(f"\n🏆 TOP 10 EQUIPOS POR PORCENTAJE DE VICTORIAS:")
print("="*80)
top_10 = df_sorted[['equipo', 'partidos_disputados', 'ganados', 'win_rate', 
                     'goles_por_partido', 'puntos_por_partido']].head(10)
print(top_10.to_string(index=False))

print(f"\n📉 BOTTOM 5 EQUIPOS POR PORCENTAJE DE VICTORIAS:")
print("="*80)
bottom_5 = df_sorted[['equipo', 'partidos_disputados', 'ganados', 'win_rate', 
                       'goles_por_partido', 'puntos_por_partido']].tail(5)
print(bottom_5.to_string(index=False))

# Estadísticas generales
print(f"\n📊 ESTADÍSTICAS GENERALES DE LA COMPETICIÓN:")
print("="*80)
print(f"Win Rate promedio: {df['win_rate'].mean():.2f}%")
print(f"Goles por partido promedio: {df['goles_por_partido'].mean():.2f}")
print(f"Disparos por partido promedio: {df['disparos_por_partido'].mean():.2f}")
print(f"Ataques por partido promedio: {df['ataques_por_partido'].mean():.2f}")
print(f"Eficiencia de disparo promedio: {df['eficiencia_disparo'].mean():.2f}%")

# Crear visualizaciones
fig = plt.figure(figsize=(18, 12))

# 1. Top 15 equipos por win_rate
ax1 = plt.subplot(3, 3, 1)
top_15 = df_sorted.head(15)
ax1.barh(range(len(top_15)), top_15['win_rate'], color='#2ecc71')
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['equipo'], fontsize=8)
ax1.set_xlabel('Win Rate (%)')
ax1.set_title('Top 15 - Porcentaje de Victorias', fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# 2. Distribución de win_rate
ax2 = plt.subplot(3, 3, 2)
ax2.hist(df['win_rate'], bins=15, color='#3498db', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Win Rate (%)')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Distribución de Win Rate', fontweight='bold')
ax2.axvline(df['win_rate'].mean(), color='red', linestyle='--', 
            label=f'Media: {df["win_rate"].mean():.1f}%')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Goles por partido vs Win Rate
ax3 = plt.subplot(3, 3, 3)
scatter = ax3.scatter(df['goles_por_partido'], df['win_rate'], 
                     c=df['win_rate'], cmap='RdYlGn', s=100, alpha=0.6)
ax3.set_xlabel('Goles por Partido')
ax3.set_ylabel('Win Rate (%)')
ax3.set_title('Goles vs Win Rate', fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Win Rate (%)')
ax3.grid(alpha=0.3)

# 4. Eficiencia de disparo vs Win Rate
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(df['eficiencia_disparo'], df['win_rate'], 
           c=df['performance_index'], cmap='viridis', s=100, alpha=0.6)
ax4.set_xlabel('Eficiencia de Disparo (%)')
ax4.set_ylabel('Win Rate (%)')
ax4.set_title('Eficiencia vs Win Rate', fontweight='bold')
ax4.grid(alpha=0.3)

# 5. Ataques por partido vs Win Rate
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(df['ataques_por_partido'], df['win_rate'], 
           c=df['goles_por_partido'], cmap='plasma', s=100, alpha=0.6)
ax5.set_xlabel('Ataques por Partido')
ax5.set_ylabel('Win Rate (%)')
ax5.set_title('Ataques vs Win Rate', fontweight='bold')
ax5.grid(alpha=0.3)

# 6. Puntos por partido
ax6 = plt.subplot(3, 3, 6)
top_points = df_sorted.head(10)
ax6.bar(range(len(top_points)), top_points['puntos_por_partido'], 
        color='#e67e22', alpha=0.7)
ax6.set_xticks(range(len(top_points)))
ax6.set_xticklabels(top_points['equipo'], rotation=45, ha='right', fontsize=8)
ax6.set_ylabel('Puntos por Partido')
ax6.set_title('Top 10 - Puntos por Partido', fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

# 7. Correlación entre variables clave
ax7 = plt.subplot(3, 3, 7)
corr_vars = ['win_rate', 'goles_por_partido', 'disparos_por_partido', 
             'ataques_por_partido', 'eficiencia_disparo', 'performance_index']
corr_matrix = df[corr_vars].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=ax7, cbar_kws={'shrink': 0.8})
ax7.set_title('Matriz de Correlación', fontweight='bold')

# 8. Performance Index vs Win Rate
ax8 = plt.subplot(3, 3, 8)
ax8.scatter(df['performance_index'], df['win_rate'], 
           s=df['goles_por_partido']*50, alpha=0.5, c=df['win_rate'], 
           cmap='RdYlGn')
ax8.set_xlabel('Performance Index')
ax8.set_ylabel('Win Rate (%)')
ax8.set_title('Performance Index vs Win Rate\n(tamaño = goles/partido)', 
              fontweight='bold')
ax8.grid(alpha=0.3)

# 9. Distribución de resultados (ganados, empates, perdidos)
ax9 = plt.subplot(3, 3, 9)
results_data = [df['ganados'].sum(), df['empates'].sum(), df['perdidos'].sum()]
ax9.pie(results_data, labels=['Ganados', 'Empates', 'Perdidos'], 
        autopct='%1.1f%%', colors=['#2ecc71', '#f39c12', '#e74c3c'],
        startangle=90)
ax9.set_title('Distribución de Resultados\n(Total Liga)', fontweight='bold')

plt.tight_layout()
plt.savefig('ucl_eda_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualización guardada: ucl_eda_analysis.png")

# Análisis de correlación detallado
print(f"\n🔍 CORRELACIONES CON WIN_RATE:")
print("="*80)
correlations = df[corr_vars].corr()['win_rate'].sort_values(ascending=False)[1:]
for var, corr in correlations.items():
    print(f"{var:30s}: {corr:6.3f}")

# =============================================================================
# FASE 5: PREPARACIÓN PARA MODELADO
# =============================================================================
print("\n" + "="*80)
print("[FASE 5] PREPARACIÓN PARA MODELADO")
print("-"*80)

# Definir características para el modelo
feature_columns = [
    'goles_por_partido',
    'disparos_por_partido',
    'ataques_por_partido',
    'eficiencia_disparo',
    'eficiencia_ataque',
    'ratio_disparos_ataques',
    'puntos_por_partido',
    'draw_rate',
    'defeat_rate',
    'performance_index'
]

# Variable objetivo
target = 'win_rate'

X = df[feature_columns]
y = df[target]

print(f"\n📊 Variables predictoras ({len(feature_columns)}):")
for i, feat in enumerate(feature_columns, 1):
    print(f"   {i:2d}. {feat}")

print(f"\n🎯 Variable objetivo: {target}")
print(f"   Rango: [{y.min():.2f}%, {y.max():.2f}%]")
print(f"   Media: {y.mean():.2f}%")
print(f"   Mediana: {y.median():.2f}%")

# Dividir en train/test
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

print(f"\n✂️ División de datos:")
print(f"   Training: {len(X_train)} equipos ({(1-test_size)*100:.0f}%)")
print(f"   Testing: {len(X_test)} equipos ({test_size*100:.0f}%)")

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_full_scaled = scaler.transform(X)

print(f"\n✓ Características estandarizadas (media=0, std=1)")

# =============================================================================
# FASE 6: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS DE REGRESIÓN
# =============================================================================
print("\n" + "="*80)
print("[FASE 6] ENTRENAMIENTO DE MODELOS DE REGRESIÓN")
print("-"*80)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = {}

print(f"\n🔄 Entrenando y evaluando modelos...\n")

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Modelo: {name}")
    print('='*60)
    
    # Entrenar
    model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Métricas
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n📊 Métricas de Entrenamiento:")
    print(f"   R² Score: {r2_train:.4f}")
    print(f"   RMSE: {rmse_train:.4f}%")
    
    print(f"\n📊 Métricas de Testing:")
    print(f"   R² Score: {r2_test:.4f}")
    print(f"   RMSE: {rmse_test:.4f}%")
    print(f"   MAE: {mae_test:.4f}%")
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                 scoring='r2')
    print(f"\n🔄 Cross-Validation (5-fold):")
    print(f"   R² Scores: {cv_scores}")
    print(f"   Media: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    results[name] = {
        'model': model,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'cv_mean': cv_scores.mean(),
        'y_pred_test': y_pred_test
    }

# Seleccionar mejor modelo
best_model_name = max(results, key=lambda x: results[x]['r2_test'])
best_model = results[best_model_name]['model']

print(f"\n\n{'='*80}")
print(f"🏆 MEJOR MODELO: {best_model_name}")
print(f"   R² Test: {results[best_model_name]['r2_test']:.4f}")
print(f"   RMSE Test: {results[best_model_name]['rmse_test']:.4f}%")
print(f"   MAE Test: {results[best_model_name]['mae_test']:.4f}%")
print('='*80)

# Visualizar resultados del mejor modelo
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predicciones vs Valores Reales (Test)
ax1 = axes[0, 0]
ax1.scatter(y_test, results[best_model_name]['y_pred_test'], alpha=0.6, s=100)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Predicción perfecta')
ax1.set_xlabel('Win Rate Real (%)')
ax1.set_ylabel('Win Rate Predicho (%)')
ax1.set_title(f'Predicciones vs Reales - {best_model_name}', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Residuos
ax2 = axes[0, 1]
residuals = y_test - results[best_model_name]['y_pred_test']
ax2.scatter(results[best_model_name]['y_pred_test'], residuals, alpha=0.6, s=100)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Win Rate Predicho (%)')
ax2.set_ylabel('Residuos (%)')
ax2.set_title('Análisis de Residuos', fontweight='bold')
ax2.grid(alpha=0.3)

# 3. Comparación de modelos (R² Test)
ax3 = axes[1, 0]
model_names = list(results.keys())
r2_scores = [results[m]['r2_test'] for m in model_names]
colors = ['#2ecc71' if m == best_model_name else '#3498db' for m in model_names]
ax3.barh(model_names, r2_scores, color=colors)
ax3.set_xlabel('R² Score')
ax3.set_title('Comparación de Modelos (R² Test)', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Importancia de características (si disponible)
ax4 = axes[1, 1]
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    ax4.barh(range(len(feat_imp)), feat_imp['importance'], color='#e67e22')
    ax4.set_yticks(range(len(feat_imp)))
    ax4.set_yticklabels(feat_imp['feature'], fontsize=9)
    ax4.set_xlabel('Importancia')
    ax4.set_title('Importancia de Características', fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
elif hasattr(best_model, 'coef_'):
    feat_imp = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': np.abs(best_model.coef_)
    }).sort_values('coefficient', ascending=False)
    
    ax4.barh(range(len(feat_imp)), feat_imp['coefficient'], color='#e67e22')
    ax4.set_yticks(range(len(feat_imp)))
    ax4.set_yticklabels(feat_imp['feature'], fontsize=9)
    ax4.set_xlabel('|Coeficiente|')
    ax4.set_title('Importancia de Características (Coeficientes)', fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Importancia de características\nno disponible para este modelo',
             ha='center', va='center', fontsize=12)
    ax4.axis('off')

plt.tight_layout()
plt.savefig('ucl_model_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualización guardada: ucl_model_evaluation.png")

# =============================================================================
# FASE 7: PREDICCIONES Y PROBABILIDADES DE VICTORIA
# =============================================================================
print("\n" + "="*80)
print("[FASE 7] PREDICCIONES DE PROBABILIDAD DE VICTORIA")
print("-"*80)

# Generar predicciones para todos los equipos
y_pred_all = best_model.predict(X_full_scaled)

# Crear DataFrame de resultados
df_predictions = pd.DataFrame({
    'Equipo': df['equipo'],
    'Win_Rate_Real': df['win_rate'],
    'Win_Rate_Predicho': y_pred_all,
    'Diferencia': y_pred_all - df['win_rate'],
    'Partidos_Jugados': df['partidos_disputados'],
    'Victorias_Actuales': df['ganados'],
    'Goles_Por_Partido': df['goles_por_partido'],
    'Performance_Index': df['performance_index'],
    'Puntos': df['puntos']
})

# Calcular probabilidad de victoria en próximo partido
# Normalizamos el win_rate predicho a una escala de 0-100%
df_predictions['Prob_Victoria_Proximo'] = np.clip(df_predictions['Win_Rate_Predicho'], 0, 100)

# Clasificar nivel de confianza
def classify_confidence(prob):
    if prob >= 70:
        return 'Muy Alta'
    elif prob >= 50:
        return 'Alta'
    elif prob >= 30:
        return 'Media'
    else:
        return 'Baja'

df_predictions['Nivel_Confianza'] = df_predictions['Prob_Victoria_Proximo'].apply(classify_confidence)

# Ordenar por probabilidad
df_predictions = df_predictions.sort_values('Prob_Victoria_Proximo', ascending=False)

print(f"\n🎯 PREDICCIONES GENERADAS PARA {len(df_predictions)} EQUIPOS")
print("="*80)

print(f"\n🏆 TOP 10 EQUIPOS CON MAYOR PROBABILIDAD DE VICTORIA:")
print("="*80)
top_10_pred = df_predictions[['Equipo', 'Win_Rate_Real', 'Prob_Victoria_Proximo', 
                               'Nivel_Confianza', 'Goles_Por_Partido']].head(10)
print(top_10_pred.to_string(index=False))

print(f"\n📉 EQUIPOS CON MENOR PROBABILIDAD DE VICTORIA:")
print("="*80)
bottom_5_pred = df_predictions[['Equipo', 'Win_Rate_Real', 'Prob_Victoria_Proximo', 
                                'Nivel_Confianza', 'Goles_Por_Partido']].tail(5)
print(bottom_5_pred.to_string(index=False))

# Estadísticas de predicciones
print(f"\n📊 ESTADÍSTICAS DE PREDICCIONES:")
print("="*80)
print(f"Probabilidad promedio: {df_predictions['Prob_Victoria_Proximo'].mean():.2f}%")
print(f"Probabilidad máxima: {df_predictions['Prob_Victoria_Proximo'].max():.2f}%")
print(f"Probabilidad mínima: {df_predictions['Prob_Victoria_Proximo'].min():.2f}%")
print(f"Desviación estándar: {df_predictions['Prob_Victoria_Proximo'].std():.2f}%")

print(f"\n📊 DISTRIBUCIÓN POR NIVEL DE CONFIANZA:")
conf_dist = df_predictions['Nivel_Confianza'].value_counts()
for nivel, count in conf_dist.items():
    pct = (count / len(df_predictions)) * 100
    print(f"   {nivel:15s}: {count:3d} equipos ({pct:.1f}%)")

# Guardar predicciones a CSV
df_predictions.to_csv('ucl_predictions_final.csv', index=False, encoding='utf-8')
print(f"\n✓ Predicciones guardadas en: ucl_predictions_final.csv")

# =============================================================================
# VISUALIZACIÓN DE PREDICCIONES
# =============================================================================
print(f"\n🎨 Generando visualizaciones de predicciones...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Top equipos por probabilidad de victoria
ax1 = axes[0, 0]
top_15_viz = df_predictions.head(15)
colors_1 = ['#2ecc71' if conf == 'Muy Alta' else '#3498db' if conf == 'Alta' else '#f39c12' 
            for conf in top_15_viz['Nivel_Confianza']]
ax1.barh(range(len(top_15_viz)), top_15_viz['Prob_Victoria_Proximo'], color=colors_1)
ax1.set_yticks(range(len(top_15_viz)))
ax1.set_yticklabels(top_15_viz['Equipo'], fontsize=9)
ax1.set_xlabel('Probabilidad de Victoria en Próximo Partido (%)')
ax1.set_title('Top 15 - Probabilidad de Victoria', fontweight='bold', fontsize=12)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Agregar valores en las barras
for i, v in enumerate(top_15_viz['Prob_Victoria_Proximo']):
    ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)

# 2. Comparación: Real vs Predicho
ax2 = axes[0, 1]
x_pos = np.arange(len(top_15_viz))
width = 0.35
ax2.bar(x_pos - width/2, top_15_viz['Win_Rate_Real'], width, 
        label='Win Rate Real', color='#3498db', alpha=0.7)
ax2.bar(x_pos + width/2, top_15_viz['Prob_Victoria_Proximo'], width,
        label='Predicción', color='#2ecc71', alpha=0.7)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(top_15_viz['Equipo'], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Porcentaje (%)')
ax2.set_title('Comparación: Real vs Predicho (Top 15)', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Distribución de probabilidades
ax3 = axes[1, 0]
ax3.hist(df_predictions['Prob_Victoria_Proximo'], bins=20, 
         color='#9b59b6', edgecolor='black', alpha=0.7)
ax3.axvline(df_predictions['Prob_Victoria_Proximo'].mean(), 
            color='red', linestyle='--', linewidth=2, 
            label=f"Media: {df_predictions['Prob_Victoria_Proximo'].mean():.1f}%")
ax3.set_xlabel('Probabilidad de Victoria (%)')
ax3.set_ylabel('Número de Equipos')
ax3.set_title('Distribución de Probabilidades de Victoria', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Scatter: Performance Index vs Probabilidad
ax4 = axes[1, 1]
scatter = ax4.scatter(df_predictions['Performance_Index'], 
                     df_predictions['Prob_Victoria_Proximo'],
                     c=df_predictions['Goles_Por_Partido'], 
                     s=df_predictions['Partidos_Jugados']*20,
                     cmap='YlOrRd', alpha=0.6, edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Performance Index')
ax4.set_ylabel('Probabilidad de Victoria (%)')
ax4.set_title('Performance vs Probabilidad\n(tamaño = partidos jugados, color = goles/partido)', 
              fontweight='bold', fontsize=11)
ax4.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Goles por Partido', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('ucl_predictions_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualización guardada: ucl_predictions_visualization.png")

# =============================================================================
# FASE 8: ANÁLISIS INTERPRETATIVO
# =============================================================================
print("\n" + "="*80)
print("[FASE 8] ANÁLISIS INTERPRETATIVO DEL MODELO")
print("-"*80)

print(f"\n📊 RESUMEN DEL MODELO PREDICTIVO")
print("="*80)
print(f"Modelo utilizado: {best_model_name}")
print(f"R² Score (Test): {results[best_model_name]['r2_test']:.4f}")
print(f"Error promedio (MAE): {results[best_model_name]['mae_test']:.2f} puntos porcentuales")
print(f"RMSE: {results[best_model_name]['rmse_test']:.2f}%")

print(f"\n🔍 VARIABLES MÁS INFLUYENTES:")
if hasattr(best_model, 'feature_importances_'):
    feat_importance = pd.DataFrame({
        'Variable': feature_columns,
        'Importancia': best_model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    print("="*80)
    for idx, row in feat_importance.iterrows():
        bar_length = int(row['Importancia'] * 50)
        bar = '█' * bar_length
        print(f"{row['Variable']:30s}: {bar} {row['Importancia']:.4f}")
    
    print(f"\n💡 INTERPRETACIÓN:")
    print("="*80)
    top_3 = feat_importance.head(3)
    print(f"Las 3 variables más importantes son:")
    for i, (idx, row) in enumerate(top_3.iterrows(), 1):
        print(f"  {i}. {row['Variable']} ({row['Importancia']:.3f})")
    
    if 'goles_por_partido' in top_3['Variable'].values:
        print(f"\n   → La eficiencia goleadora es CRÍTICA para predecir victorias")
    if 'eficiencia_disparo' in top_3['Variable'].values:
        print(f"   → La conversión de disparos en goles es un factor determinante")
    if 'performance_index' in top_3['Variable'].values:
        print(f"   → El rendimiento general compuesto es altamente predictivo")

elif hasattr(best_model, 'coef_'):
    coef_df = pd.DataFrame({
        'Variable': feature_columns,
        'Coeficiente': best_model.coef_,
        'Impacto_Abs': np.abs(best_model.coef_)
    }).sort_values('Impacto_Abs', ascending=False)
    
    print("="*80)
    for idx, row in coef_df.iterrows():
        direction = "↑" if row['Coeficiente'] > 0 else "↓"
        bar_length = int(row['Impacto_Abs'] * 20)
        bar = '█' * max(1, bar_length)
        print(f"{row['Variable']:30s}: {direction} {bar} {row['Coeficiente']:7.3f}")
    
    print(f"\n💡 INTERPRETACIÓN:")
    print("="*80)
    print(f"Variables con impacto POSITIVO en probabilidad de victoria:")
    positive = coef_df[coef_df['Coeficiente'] > 0].head(3)
    for i, (idx, row) in enumerate(positive.iterrows(), 1):
        print(f"  {i}. {row['Variable']} (coef: +{row['Coeficiente']:.3f})")
    
    if len(coef_df[coef_df['Coeficiente'] < 0]) > 0:
        print(f"\nVariables con impacto NEGATIVO:")
        negative = coef_df[coef_df['Coeficiente'] < 0].head(2)
        for i, (idx, row) in enumerate(negative.iterrows(), 1):
            print(f"  {i}. {row['Variable']} (coef: {row['Coeficiente']:.3f})")

# Análisis de errores
print(f"\n📊 ANÁLISIS DE PRECISIÓN POR GRUPOS:")
print("="*80)

# Agrupar equipos por nivel de win_rate real
df_predictions['Categoria_Real'] = pd.cut(df_predictions['Win_Rate_Real'],
                                          bins=[0, 25, 50, 75, 100],
                                          labels=['Bajo (0-25%)', 'Medio (25-50%)', 
                                                 'Alto (50-75%)', 'Muy Alto (75-100%)'])

error_by_category = df_predictions.groupby('Categoria_Real').agg({
    'Diferencia': ['mean', 'std', 'count']
}).round(2)

print("\nError promedio por categoría de rendimiento:")
print(error_by_category)

# Identificar mejores y peores predicciones
print(f"\n🎯 PREDICCIONES MÁS PRECISAS:")
print("="*80)
df_predictions['Error_Abs'] = np.abs(df_predictions['Diferencia'])
best_predictions = df_predictions.nsmallest(5, 'Error_Abs')[
    ['Equipo', 'Win_Rate_Real', 'Prob_Victoria_Proximo', 'Error_Abs']]
print(best_predictions.to_string(index=False))

print(f"\n⚠️ PREDICCIONES CON MAYOR DESVIACIÓN:")
print("="*80)
worst_predictions = df_predictions.nlargest(5, 'Error_Abs')[
    ['Equipo', 'Win_Rate_Real', 'Prob_Victoria_Proximo', 'Diferencia']]
print(worst_predictions.to_string(index=False))

# =============================================================================
# RECOMENDACIONES Y CONCLUSIONES
# =============================================================================
print("\n" + "="*80)
print("[RECOMENDACIONES ESTRATÉGICAS]")
print("-"*80)

# Equipos de alto potencial
high_potential = df_predictions[
    (df_predictions['Prob_Victoria_Proximo'] >= 60) & 
    (df_predictions['Partidos_Jugados'] >= 3)
]

print(f"\n🌟 EQUIPOS DE ALTO POTENCIAL (≥60% probabilidad):")
print("="*80)
if len(high_potential) > 0:
    print(f"   {len(high_potential)} equipos identificados como favoritos")
    for idx, row in high_potential.iterrows():
        print(f"   • {row['Equipo']:25s} → {row['Prob_Victoria_Proximo']:.1f}% "
              f"(Goles/partido: {row['Goles_Por_Partido']:.2f})")
else:
    print("   No hay equipos con probabilidad ≥60% en este análisis")

# Equipos en riesgo
at_risk = df_predictions[
    (df_predictions['Prob_Victoria_Proximo'] < 30) & 
    (df_predictions['Partidos_Jugados'] >= 3)
]

print(f"\n⚠️ EQUIPOS EN RIESGO (<30% probabilidad):")
print("="*80)
if len(at_risk) > 0:
    print(f"   {len(at_risk)} equipos necesitan mejorar su rendimiento")
    for idx, row in at_risk.iterrows():
        print(f"   • {row['Equipo']:25s} → {row['Prob_Victoria_Proximo']:.1f}% "
              f"(Goles/partido: {row['Goles_Por_Partido']:.2f})")
else:
    print("   No hay equipos en riesgo crítico en este análisis")

# Sorpresas potenciales (equipos superando expectativas)
df_predictions['Sobre_Expectativa'] = (
    df_predictions['Prob_Victoria_Proximo'] - df_predictions['Win_Rate_Real']
)
surprises = df_predictions.nlargest(5, 'Sobre_Expectativa')[
    ['Equipo', 'Win_Rate_Real', 'Prob_Victoria_Proximo', 'Sobre_Expectativa']
]

print(f"\n🚀 SORPRESAS POSITIVAS (superan expectativas):")
print("="*80)
print(surprises.to_string(index=False))

print(f"\n" + "="*80)
print("[LIMITACIONES Y CONSIDERACIONES]")
print("-"*80)

print(f"\n⚠️ LIMITACIONES DEL MODELO:")
print(f"   • Basado únicamente en estadísticas agregadas de equipos")
print(f"   • No considera enfrentamientos directos específicos")
print(f"   • No incluye lesiones, suspensiones o rotaciones")
print(f"   • No contempla factor psicológico o motivacional")
print(f"   • Asume que el rendimiento pasado predice el futuro")
print(f"   • Cantidad limitada de partidos por equipo ({df['partidos_disputados'].mean():.1f} promedio)")

print(f"\n💡 RECOMENDACIONES PARA MEJORAR:")
print(f"   1. Incorporar datos de enfrentamientos directos (H2H)")
print(f"   2. Incluir estadísticas defensivas (goles recibidos, etc.)")
print(f"   3. Añadir análisis de forma reciente (últimos 3-5 partidos)")
print(f"   4. Considerar contexto del partido (local/visitante, fase)")
print(f"   5. Integrar datos de jugadores clave y su disponibilidad")
print(f"   6. Actualizar modelo continuamente con nuevos resultados")

# =============================================================================
# EXPORTACIÓN FINAL
# =============================================================================
print("\n" + "="*80)
print("[EXPORTACIÓN DE RESULTADOS]")
print("-"*80)

# Crear resumen ejecutivo
summary_stats = {
    'Modelo': best_model_name,
    'R2_Score': results[best_model_name]['r2_test'],
    'MAE': results[best_model_name]['mae_test'],
    'RMSE': results[best_model_name]['rmse_test'],
    'Equipos_Analizados': len(df_predictions),
    'Prob_Victoria_Promedio': df_predictions['Prob_Victoria_Proximo'].mean(),
    'Equipos_Alta_Confianza': len(df_predictions[df_predictions['Nivel_Confianza'] == 'Muy Alta']),
    'Equipos_Riesgo': len(at_risk),
    'Fecha_Analisis': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('ucl_model_summary.csv', index=False, encoding='utf-8')

# Crear archivo de importancia de variables
if hasattr(best_model, 'feature_importances_'):
    feat_importance.to_csv('ucl_feature_importance.csv', index=False, encoding='utf-8')
elif hasattr(best_model, 'coef_'):
    coef_df.to_csv('ucl_feature_coefficients.csv', index=False, encoding='utf-8')

print(f"\n✅ ARCHIVOS GENERADOS:")
print("="*80)
print(f"   📄 ucl_predictions_final.csv - Predicciones detalladas")
print(f"   📄 ucl_model_summary.csv - Resumen ejecutivo del modelo")
if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
    print(f"   📄 ucl_feature_importance.csv - Importancia de variables")
print(f"   📊 ucl_eda_analysis.png - Análisis exploratorio")
print(f"   📊 ucl_model_evaluation.png - Evaluación del modelo")
print(f"   📊 ucl_predictions_visualization.png - Visualización de predicciones")

print("\n" + "="*80)
print("✅ PROCESO COMPLETADO EXITOSAMENTE")
print("="*80)

print(f"\n📊 ESTADÍSTICAS FINALES:")
print(f"   • Equipos analizados: {len(df_predictions)}")
print(f"   • Precisión del modelo (R²): {results[best_model_name]['r2_test']:.4f}")
print(f"   • Error promedio: ±{results[best_model_name]['mae_test']:.2f} puntos porcentuales")
print(f"   • Equipo más probable de ganar: {df_predictions.iloc[0]['Equipo']} "
      f"({df_predictions.iloc[0]['Prob_Victoria_Proximo']:.1f}%)")

print(f"\n🎯 PRÓXIMOS PASOS:")
print(f"   1. Revisar 'ucl_predictions_final.csv' para ver todas las predicciones")
print(f"   2. Analizar las visualizaciones generadas (.png)")
print(f"   3. Comparar predicciones con resultados reales futuros")
print(f"   4. Re-entrenar el modelo cuando haya más partidos disputados")
print(f"   5. Considerar las recomendaciones de mejora mencionadas")

print("\n" + "="*80)
print("Gracias por usar el Modelo Predictivo UEFA Champions League")
print("Desarrollado con Python, scikit-learn y análisis estadístico avanzado")
print("="*80 + "\n")