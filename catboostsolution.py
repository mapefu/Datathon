from catboost import Pool, CatBoostClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm


data = pd.read_csv("data_train_proces_temporadas.csv", sep=";")
test = pd.read_csv("test/grouped_data_test_temporadas.csv", sep=";")
print('1')


Xtest = test.drop(columns=["ID"])
X = data.drop(columns=["ID", "Production", "weekly_demand"])
y = data['weekly_demand']
print('2')

columnas_categoricas = list(data.columns[1:20]) + [data.columns[23]]
print('3')

X[columnas_categoricas] = X[columnas_categoricas].fillna("nan").astype(str)
Xtest[columnas_categoricas] = Xtest[columnas_categoricas].fillna("nan").astype(str)
print('4')

train_data = Pool(X, y, cat_features=columnas_categoricas)
test_pool = Pool(Xtest, cat_features=columnas_categoricas)
print('5')

modelo = CatBoostClassifier(iterations=1, learning_rate=0.1, depth=2, eval_metric='Accuracy')
print('6')

modelo.fit(train_data, verbose=0)
print('7')

y_pred = modelo.predict(test_pool)
print('8')

factor = 1.13
predicciones_aumentadas = y_pred * factor
print('9')

predicciones_int = np.round(predicciones_aumentadas).astype(int)


resultados = pd.DataFrame({
    "ID": test["ID"].values,
    "Production": predicciones_int
})


resultados.to_csv("submission.csv", sep=",", index=False)
