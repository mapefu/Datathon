from catboost import Pool, CatBoostClassifier


data = pd.read_csv('data_train_proces_temporadas.csv')
test = pd.read_csv('test\\grouped_data_test_temporadas.csv')

Xtest = test.drop("ID", "Production", "weekly_demand", axis=1)
X = data.drop("ID", "Production", "weekly_demand", axis=1)
y = data['weekly_demand']

columnas_categoricas = list(data.columns[1:20]) + list(data.columns[23])

train_data = Pool(X, y, cat_features=columnas_categoricas)
test_pool = Pool(X_test, cat_features=cat_features)

modelo = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, eval_metric='Accuracy')
modelo.fit(train_data, verbose=100)

y_pred = model.predict(test_pool)

factor = 1.13
predicciones_aumentadas = y_pred * factor

predicciones_int = np.round(predicciones_aumentadas).astype(int)

resultados = pd.DataFrame({
    "ID": test["ID"].values,
    "Production": predicciones_int
})

resultados.to_csv("submission.csv", sep=",", index=False)

