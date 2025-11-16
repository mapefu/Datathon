import pandas as pd

df = pd.read_csv("grouped_data.csv", sep=";")

# Busquem columnes categòriques
cat_cols = df.select_dtypes(include=["object"]).columns

# Fem Label Encoding per columna
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# Guardem el resultat
df.to_csv("grouped_data_encoded.csv", sep=";", index=False)

print("✔️ Conversió completada: strings → ints")