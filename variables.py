import pandas as pd

# Leer CSV indicando que el separador es ;
df = pd.read_csv('dades\\dades.csv', sep=';')

col = pd.read_csv('train.csv', sep=';', usecols=[df.columns[0], df.columns[27], df.columns[-1]])

families = []

for n in df.columns[2:]:
    families.append(df[n].unique())

families.to_csv('families.csv', sep=';', index=False)

print(families)