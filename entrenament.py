import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar datos (archivo CSV)
df = pd.read_csv('grouped_data.csv', sep = ';')

# Definir variables independientes (X) y variable a predecir (y)
X = df[[df.columns[1], df.columns[3], df.columns[-2]]]  # ejemplo
y = df[df.columns[2]]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear y entrenar modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Hacer predicciones
predicciones = modelo.predict(X_test)

print(predicciones)