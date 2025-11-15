import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

df = pd.read_csv('train.csv', sep=';')

X = df[list(df.columns[1:8]) + [df.columns[18], df.columns[9], df.columns[10]] + list(df.columns[20:28])]
y = df[df.columns[-2]]

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

desviacion_tipica = np.std(resultados['Valor Real'] - resultados['Predicción'])
print(f"Desviación típica entre predicción y valor real: {desviacion_tipica}")

# Guardar en archivo CSV
resultados.to_csv('predicciones_y_reales.csv', index=False, sep=';')
