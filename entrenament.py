import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('data_train_proces_temporadas.csv', sep=';')

for i in range(5):
    X = df[list(df.columns[1:5]) + list(df.columns[6:18]) + list(df.columns[20:-2])]
    y = df[df.columns[-1]]

    X_encoded = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.15)

    modelo = RandomForestRegressor()
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    # Crear DataFrame con resultados reales y predicciones
    resultados = pd.DataFrame({
        'Valor Real': y_test.values,
        'Predicción': predicciones
    })

    r2 = r2_score(y_test, predicciones)

    print(f"{i + 1}-R^2: {r2}")