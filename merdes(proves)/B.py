import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. LLEGIR DADES
df = pd.read_csv('data_train_proces_temporadas.csv', sep=';')
dt = pd.read_csv('test\\grouped_data_test_temporadas.csv', sep=';')

# 2. DEFINIR FEATURES I TARGET DE MANERA MÉS CLARA
# Columnes que tenim al train:
# ['ID', 'id_season', 'family', 'category', 'fabric', 'color_rgb',
#  'length_type', 'silhouette_type', 'waist_type', 'neck_lapel_type',
#  'sleeve_length_type', 'heel_shape_type', 'toecap_type', 'woven_structure',
#  'knit_structure', 'print_type', 'archetype', 'moment', 'phase_in',
#  'phase_out', 'life_cycle_length', 'num_stores', 'num_sizes',
#  'has_plus_sizes', 'price', 'Production', 'weekly_demand']

target_col = "weekly_demand"

# No volem fer servir ID ni la pròpia target ni Production com a feature
cols_to_exclude = ["ID", "Production", "weekly_demand"]

feature_cols = [c for c in df.columns if c not in cols_to_exclude]

print("Features utilitzades:")
print(feature_cols)
print()

X = df[feature_cols].copy()
y = df[target_col].copy()

Xt = dt[feature_cols].copy()

# Assegurem que has_plus_sizes és numèrica (0/1) si existeix
if "has_plus_sizes" in X.columns:
    X["has_plus_sizes"] = X["has_plus_sizes"].astype(int)
    Xt["has_plus_sizes"] = Xt["has_plus_sizes"].astype(int)

# 3. ONE-HOT ENCODING (CATEGÒRIQUES → DUMMIES)
X_train = pd.get_dummies(X)
X_test = pd.get_dummies(Xt)

# Alineem columnes: test ha de tenir EXACTAMENT les mateixes que train
X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Forma X_train:", X_train.shape)
print("Forma X_test_aligned:", X_test_aligned.shape)
print()

# 4. VALIDACIÓ RÀPIDA PER VEURE COM DE BO ÉS EL MODEL
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

# Random Forest una mica més potent i estable
modelo = RandomForestRegressor(
    n_estimators=400,      # més arbres → més estabilitat
    max_depth=20,         # evita arbres massa profunds i sorollosos
    min_samples_leaf=2,   # fulles amb com a mínim 2 mostres
    n_jobs=-1,            # usa tots els nuclis de CPU
    random_state=42       # resultats repetibles
)

print("Entrenant model per validació interna...")
modelo.fit(X_tr, y_tr)

y_val_pred = modelo.predict(X_val)

mae = mean_absolute_error(y_val, y_val_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

print("Resultats validació (20% train):")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")
print()

# 5. REENTRENAR AMB TOTES LES DADES D'ENTRENAMENT
print("Reentrenant amb totes les dades d'entrenament...")
modelo.fit(X_train, y)

# 6. PREDIR SOBRE EL TEST
print("Predint sobre el conjunt de test...")
predicciones = modelo.predict(X_test_aligned)

# Opcional: multiplicar per un factor per evitar quedarte curt de producció
factor = 1.11
predicciones_aumentadas = predicciones * factor

# Arrodonir i convertir a enters
predicciones_int = np.round(predicciones_aumentadas).astype(int)

# 7. CREAR DATAFRAME DE RESULTATS
resultados = pd.DataFrame({
    'ID': dt['ID'].values,
    'Production': predicciones_int
})

# 8. GUARDAR FITXER DE SUBMISSIÓ
resultados.to_csv('submission_rf.csv', sep=',', index=False)
print("✔️ Fitxer 'submission_rf.csv' creat (ID,Production)")
