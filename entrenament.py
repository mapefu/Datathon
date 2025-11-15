import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('grouped_data.csv', sep=';')

X = df[[df.columns[1], df.columns[3], df.columns[-2]]]
y = df[df.columns[2]]

# Convertir las columnas categóricas a variables numéricas
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

print(predicciones)
