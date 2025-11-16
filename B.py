import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint, uniform

# ---------------- 1. LLEGIR DADES ----------------
train_path = "data_train_proces_temporadas.csv"
test_path  = "test/grouped_data_test_temporadas.csv"

print(f"Llegint train de: {train_path}")
df = pd.read_csv(train_path, sep=";")

print(f"Llegint test de : {test_path}")
dt = pd.read_csv(test_path, sep=";")

print("Columnes train:", df.columns.tolist())
print("Columnes test :", dt.columns.tolist())
print()

# ---------------- 2. DEFINIR FEATURES I TARGET ----------------
# Target = demanda total (weekly_demand)
target_col = "weekly_demand"

# Columnes que NO farem servir com a features
cols_to_exclude = ["ID", "Production", "weekly_demand"]

feature_cols = [c for c in df.columns if c not in cols_to_exclude]

print("Features utilitzades:")
print(feature_cols)
print()

X = df[feature_cols].copy()
y = df[target_col].copy()

Xt = dt[feature_cols].copy()

# Assegurem que booleans passen a 0/1
if "has_plus_sizes" in X.columns:
    X["has_plus_sizes"] = X["has_plus_sizes"].astype(int)
    Xt["has_plus_sizes"] = Xt["has_plus_sizes"].astype(int)

# ---------------- 3. ONE-HOT ENCODING COHERENT ----------------
# Truquet: concatenem train + test per fer dummies amb les mateixes categories
X_all = pd.concat([X, Xt], axis=0, ignore_index=True)

X_all_enc = pd.get_dummies(X_all)

# Separem de nou
X_enc  = X_all_enc.iloc[:len(X), :].copy()
Xt_enc = X_all_enc.iloc[len(X):, :].copy()

print("Forma X_enc (train):", X_enc.shape)
print("Forma Xt_enc (test) :", Xt_enc.shape)
print()

# ---------------- 4. TRAIN/VALIDATION SPLIT ----------------
# Fem una validació interna per optimitzar el model
X_tr, X_val, y_tr, y_val = train_test_split(
    X_enc, y, test_size=0.2, random_state=42
)

print("Mides:")
print("  X_tr:", X_tr.shape, "  y_tr:", y_tr.shape)
print("  X_val:", X_val.shape, " y_val:", y_val.shape)
print()

# ---------------- 5. DEFINIR RANDOM FOREST BASE ----------------
rf_base = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# ---------------- 6. ESPAI D'HIPERPARÀMETRES PER CERCA ALEATÒRIA ----------------
param_dist = {
    "n_estimators": randint(200, 800),       # nombre d'arbres
    "max_depth": randint(8, 30),            # profunditat màxima
    "min_samples_split": randint(2, 10),    # mínim mostres per fer un split
    "min_samples_leaf": randint(1, 6),      # mínim mostres a una fulla
    "max_features": ["sqrt", "log2", 0.5],  # quantes features mira cada arbre
    "bootstrap": [True, False]              # bootstrapping o no
}

# ---------------- 7. CERCA D'HIPERPARÀMETRES (RandomizedSearchCV) ----------------
# Temps no és problema → podem deixar n_iter més alt
n_iter_search = 30

print("Iniciant cerca aleatòria d'hiperparàmetres...")
random_search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    scoring="neg_mean_absolute_error",  # volem minimitzar MAE
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_tr, y_tr)

print("\nMillors hiperparàmetres trobats:")
print(random_search.best_params_)
print()

best_model = random_search.best_estimator_

# ---------------- 8. AVALUACIÓ SOBRE VALIDACIÓ ----------------
y_val_pred = best_model.predict(X_val)

mae = mean_absolute_error(y_val, y_val_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

print("RESULTATS VALIDACIÓ INTERN (amb best_model):")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")
print()

# ---------------- 9. REENTRENAR AMB TOT EL TRAIN ----------------
print("Reentrenant best_model amb totes les dades de train...")
best_model.fit(X_enc, y)

# ---------------- 10. PREDIR SOBRE TEST ----------------
print("Predint sobre Xt_enc (test final)...")
pred_test = best_model.predict(Xt_enc)

# "Inflar" una mica la producció (molt útil si penalitzen vendes perdudes)
factor = 1.15  # pots provar 1.10, 1.15, 1.20
pred_test_adj = pred_test * factor

pred_test_int = np.round(pred_test_adj).astype(int)

# ---------------- 11. CREAR FITXER DE SUBMISSIÓ ----------------
submission = pd.DataFrame({
    "ID": dt["ID"].values,
    "Production": pred_test_int
})

output_file = "submission_rf_optim.csv"
submission.to_csv(output_file, sep=",", index=False)

print(f"✔️ Fitxer '{output_file}' creat amb columnes (ID, Production)")
