from catboost import Pool, CatBoostRegressor 
import pandas as pd
import numpy as np


data = pd.read_csv("data_train_proces_temporadas.csv", sep=";")
test = pd.read_csv("test/grouped_data_test_temporadas.csv", sep=";")

Xtest = test.drop(columns=["ID", 'price'])
X = data.drop(columns=["ID", "Production", "weekly_demand", 'price']) 
y = data['weekly_demand']

columnas_categoricas = [col for col in X.columns if col in list(data.columns[1:20]) + [data.columns[23]]]


X[columnas_categoricas] = X[columnas_categoricas].fillna("nan").astype(str)
Xtest[columnas_categoricas] = Xtest[columnas_categoricas].fillna("nan").astype(str)

train_data = Pool(X, y, cat_features=columnas_categoricas)
test_pool = Pool(Xtest, cat_features=columnas_categoricas)

modelo = CatBoostRegressor(iterations=5000, learning_rate=0.05, depth=10, eval_metric='R2')
'''
print("Columnas categóricas:", columnas_categoricas)
print("Columnas en X:", X.columns.tolist())
'''
modelo.fit(train_data, verbose=100)

y_pred = modelo.predict(test_pool)

factor = 1
predicciones_aumentadas = y_pred * factor

predicciones_int = np.round(predicciones_aumentadas).astype(int)


resultados = pd.DataFrame({
    "ID": test["ID"].values,
    "Production": predicciones_int
})


resultados.to_csv("submission.csv", sep=",", index=False)
