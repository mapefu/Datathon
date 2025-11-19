from catboost import Pool, CatBoostClassifier
import pandas as pd
import numpy as np


data = pd.read_csv('data_train_proces_temporadas.csv', sep = ';')
test = pd.read_csv('test/grouped_data_test_temporadas.csv', sep = ';')

Xtest = test.drop(["ID"])
X = data.drop(["ID", "Production", "weekly_demand"])
y = data['weekly_demand']

columnas_categoricas = list(data.columns[1:20]) + [data.columns[23]]

train_data = Pool(X, y, cat_features=columnas_categoricas)
test_pool = Pool(Xtest, cat_features=columnas_categoricas)

modelo = CatBoostClassifier(iterations=10, learning_rate=0.1, depth=3, eval_metric='Accuracy')
modelo.fit(train_data, verbose=10)

y_pred = modelo.predict(test_pool)

factor = 1.13
predicciones_aumentadas = y_pred * factor

predicciones_int = np.round(predicciones_aumentadas).astype(int)

resultados = pd.DataFrame({
    "ID": test["ID"].values,
    "Production": predicciones_int
})

resultados.to_csv("submission.csv", sep=",", index=False)

