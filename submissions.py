import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('data_train_proces_temporadas.csv', sep=';')
dt = pd.read_csv('test\\grouped_data_test_temporadas.csv', sep=';')

X = df[list(df.columns[1:5])+list(df.columns[6:18])+ list(df.columns[20:-3])]
y_train = df[df.columns[-1]]
Xt = dt[list(dt.columns[1:5])+list(dt.columns[6:18])+ list(dt.columns[20:-3])]
# Convertir las columnas categóricas a variables numéricas
X_train = pd.get_dummies(X)
X_test = pd.get_dummies(Xt)

X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)

print(X_train)

#X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.1)

modelo = RandomForestRegressor()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test_aligned)

predicciones_aumentadas = (predicciones * 1.11)
# Redondear predicciones y convertir a enteros
predicciones_int = np.round(predicciones_aumentadas).astype(int)

# Crear DataFrame con la primera columna de Xt y y_test
resultados = pd.DataFrame({
    'ID': dt.iloc[:, 0].values,
    'Production': predicciones_int
})

# Guardar DataFrame a CSV
resultados.to_csv('submition.csv', sep=',', index=False)

'''
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
'''
