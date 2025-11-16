import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#LLEGIR DADES ----------------
df = pd.read_csv("data_train_proces_temporadas.csv", sep=";")
dt = pd.read_csv("test/grouped_data_test_temporadas.csv", sep=";")

'''
print("Columnes train:", df.columns.tolist())
print("Columnes test :", dt.columns.tolist())
print()
'''

#2. DEFINIR FEATURES I TARGET ----------------
# Farem servir com a target la demanda total (weekly_demand)
target_col = "weekly_demand"

#Columnes que NO volem com a features
cols_to_exclude = ["ID", "Production", "weekly_demand"]  # també podem excloure color_rgb si molesta

feature_cols = [c for c in df.columns if c not in cols_to_exclude]

print("Features que farem servir:")
print(feature_cols)
print()

X = df[feature_cols]
y = df[target_col]

Xt = dt[feature_cols]  # test té les mateixes columnes menys target

#3. ENCODING CATEGÒRIQUES (ONE-HOT) ----------------
X_enc = pd.get_dummies(X)
Xt_enc = pd.get_dummies(Xt)

#Alignem columnes: perquè test tingui exactament les mateixes que train
Xt_aligned = Xt_enc.reindex(columns=X_enc.columns, fill_value=0)

print("Forma X_enc (train):", X_enc.shape)
print("Forma Xt_aligned (test):", Xt_aligned.shape)
print()

#4. VALIDACIÓ INTERNA (TRAIN/VAL SPLIT) ----------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_enc, y, test_size=0.2, random_state=42
)

modelo = RandomForestRegressor(
    n_estimators=500,
    max_depth=22,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

modelo.fit(X_tr, y_tr)

y_val_pred = modelo.predict(X_val)

# mae = mean_absolute_error(y_val, y_val_pred)
# rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)

# print("RESULTATS VALIDACIÓ INTERNA (20% del train):")
# print(f"  MAE  : {mae:.4f}")
# print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")
# print()

#5. REENTRENAR AMB TOT EL TRAIN ----------------
modelo_full = RandomForestRegressor(
    n_estimators=500,
    max_depth=22,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

modelo_full.fit(X_enc, y)

#6. PREDIR SOBRE EL TEST ----------------
predicciones = modelo_full.predict(Xt_aligned)

#"Inflar" una mica la producció (per penalització de vendes perdudes)
factor = 1.1
predicciones_aumentadas = predicciones * factor

#Arrodonim i passem a enters
predicciones_int = np.round(predicciones_aumentadas).astype(int)

#7. CREAR CSV DE SUBMISSIÓ ----------------
resultados = pd.DataFrame({
    "ID": dt["ID"].values,
    "Production": predicciones_int
})

resultados.to_csv("submission.csv", sep=",", index=False)

print("✔️ Fitxer 'submission.csv' creat amb columnes (ID, Production)")

