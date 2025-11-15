import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('grouped_data.csv', sep=';')

X = df[[df.columns[1]] + list(df.columns[3:19])+ list(df.columns[21:-2])]
y = df[df.columns[2]]

# Convertir las columnas categóricas a variables numéricas
X_encoded = pd.get_dummies(X)

print(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.1)

modelo = RandomForestRegressor()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

# Crear DataFrame con resultados reales y predicciones
resultados = pd.DataFrame({
    'Valor Real': y_test.values,
    'Predicción': predicciones
})

mae = mean_absolute_error(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predicciones)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# Guardar en archivo CSV
resultados.to_csv('predicciones_y_reales.csv', index=False, sep=';')
