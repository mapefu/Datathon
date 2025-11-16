import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CARGA Y PREPARACIÓN DE DATOS
# ========================================
df = pd.read_csv('data_train_proces_temporadas.csv', sep=';')
dt = pd.read_csv('test\\grouped_data_test_temporadas.csv', sep=';')

print("=" * 70)
print("CARGA DE DATOS Y ANÁLISIS")
print("=" * 70)
print(f"Train shape: {df.shape}")
print(f"Test shape: {dt.shape}")

# Variable objetivo
y = df['weekly_demand']

# ========================================
# SELECCIÓN DE FEATURES (SIN PRODUCTION)
# ========================================
# Usar SOLO variables que existen en ambos datasets
features_seleccionados = [
    # Características de producto
    'id_season', 'family', 'category', 'fabric', 'color_rgb',
    # Características de tipo/forma
    'length_type', 'silhouette_type', 'waist_type', 'neck_lapel_type',
    'sleeve_length_type', 'woven_structure', 'knit_structure',
    # Características de diseño
    'print_type', 'archetype', 'moment',
    # Características temporales
    'phase_in', 'phase_out', 'life_cycle_length',
    # Variables numéricas importantes (disponibles en test)
    'num_stores', 'num_sizes', 'has_plus_sizes', 'price'
]

X = df[features_seleccionados].copy()
Xt = dt[features_seleccionados].copy()

# ========================================
# CREAR FEATURES DERIVADAS
# ========================================
# Estas características se crean a partir de variables existentes
# para compensar la pérdida de 'Production'

print("\nCreando features derivadas...")

# Feature 1: Ratio de precio a tiendas (distribución de precio)
X['price_per_store'] = X['price'] / (X['num_stores'] + 1)
Xt['price_per_store'] = Xt['price'] / (Xt['num_stores'] + 1)

# Feature 2: Capacidad total (tiendas × tallas)
X['store_size_capacity'] = X['num_stores'] * X['num_sizes']
Xt['store_size_capacity'] = Xt['num_stores'] * Xt['num_sizes']

# Feature 3: Si es premium (precio alto)
price_75_percentile = df['price'].quantile(0.75)
X['is_premium'] = (X['price'] > price_75_percentile).astype(int)
Xt['is_premium'] = (Xt['price'] > price_75_percentile).astype(int)

# Feature 4: Variedad de tallas (ordinal)
X['size_variety'] = X['num_sizes'].apply(lambda x: 0 if x <= 3 else (1 if x <= 6 else 2))
Xt['size_variety'] = Xt['num_sizes'].apply(lambda x: 0 if x <= 3 else (1 if x <= 6 else 2))

# Feature 5: Distribución (alto si muchas tiendas, bajo si pocas)
stores_75_percentile = df['num_stores'].quantile(0.75)
stores_25_percentile = df['num_stores'].quantile(0.25)
X['store_distribution'] = X['num_stores'].apply(
    lambda x: 0 if x <= stores_25_percentile else (1 if x <= stores_75_percentile else 2)
)
Xt['store_distribution'] = Xt['num_stores'].apply(
    lambda x: 0 if x <= stores_25_percentile else (1 if x <= stores_75_percentile else 2)
)

print("✓ 5 features derivadas creadas")

# ========================================
# MANEJO DE VALORES FALTANTES
# ========================================
print("\nManejo de valores faltantes...")

# Rellenar valores faltantes en variables categóricas
categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    X[col] = X[col].fillna('Unknown')
    Xt[col] = Xt[col].fillna('Unknown')

# Rellenar valores numéricos faltantes con 0
X = X.fillna(0)
Xt = Xt.fillna(0)

print(f"✓ Valores faltantes manejados")
print(f"  Train: {X.isnull().sum().sum()} valores nulos")
print(f"  Test: {Xt.isnull().sum().sum()} valores nulos")

# ========================================
# ENCODING DE VARIABLES CATEGÓRICAS
# ========================================
print("\nEncoding de variables categóricas...")

X_encoded = pd.get_dummies(X, drop_first=False)
X_test = pd.get_dummies(Xt, drop_first=False)

# Alinear columnas entre train y test
missing_cols = set(X_encoded.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_encoded.columns]

print(f"✓ Encoding completado")
print(f"  Features finales: {X_encoded.shape[1]}")

# ========================================
# ENTRENAMIENTO CON MÚLTIPLES MODELOS
# ========================================
print("\n" + "=" * 70)
print("ENTRENAMIENTO DE MODELOS")
print("=" * 70)

modelos = []
r2_scores = []
mae_scores = []
nombres_modelos = []

# Configuraciones optimizadas
configs = [
    {
        'nombre': 'RandomForest Estándar',
        'modelo': RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    },
    {
        'nombre': 'RandomForest Profundo',
        'modelo': RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    },
    {
        'nombre': 'GradientBoosting (learning=0.05)',
        'modelo': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )
    },
    {
        'nombre': 'GradientBoosting (learning=0.1)',
        'modelo': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            subsample=0.7,
            random_state=42
        )
    }
]

# Entrenar cada modelo con validación
for config in configs:
    print(f"\n{config['nombre']}...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, test_size=0.15, random_state=42
    )
    
    model = config['modelo']
    model.fit(X_train, y_train)
    
    # Predicciones en validación
    y_pred = model.predict(X_val)
    
    # Métricas
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    modelos.append(model)
    r2_scores.append(r2)
    mae_scores.append(mae)
    nombres_modelos.append(config['nombre'])
    
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

# ========================================
# SELECCIONAR MEJOR MODELO
# ========================================
mejor_idx = np.argmax(r2_scores)
mejor_r2 = r2_scores[mejor_idx]
mejor_nombre = nombres_modelos[mejor_idx]
mejor_modelo = modelos[mejor_idx]

print("\n" + "=" * 70)
print(f"✓ MEJOR MODELO: {mejor_nombre}")
print(f"  R² Final: {mejor_r2:.4f}")
print("=" * 70)

# ========================================
# PREDICCIÓN EN DATOS DE TEST
# ========================================
print("\nGenerando predicciones en test set...")

predicciones = mejor_modelo.predict(X_test)

# Asegurar valores positivos (demanda no puede ser negativa)
predicciones = np.maximum(predicciones, 0)

# Aplicar factor de ajuste opcional (basado en análisis del error)
factor_ajuste = 1.08

predicciones_ajustadas = predicciones * factor_ajuste

# Redondear a enteros
predicciones_int = np.round(predicciones_ajustadas).astype(int)

# ========================================
# GUARDAR RESULTADOS
# ========================================
# Crear DataFrame con resultados
resultados = pd.DataFrame({
    'ID': dt.iloc[:, 0].values,
    'weekly_demand': predicciones_int
})

resultados.to_csv('submissions_mejorado.csv', sep=',', index=False)
print(f"\n✓ Resultados guardados en 'submissions_mejorado.csv'")
print(f"  {len(resultados)} predicciones generadas")
print(f"  Rango de predicciones: {predicciones_int.min()} - {predicciones_int.max()}")

# ========================================
# ANÁLISIS DE IMPORTANCIA DE FEATURES
# ========================================
print("\n" + "=" * 70)
print("FEATURES MÁS IMPORTANTES (Top 20)")
print("=" * 70)

if hasattr(mejor_modelo, 'feature_importances_'):
    importancias = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importancia': mejor_modelo.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    print(f"\n{'Rank':<6} {'Feature':<40} {'Importancia':<12}")
    print("-" * 60)
    for i, (idx, row) in enumerate(importancias.head(20).iterrows(), 1):
        print(f"{i:<6} {row['Feature']:<40} {row['Importancia']:.6f}")
