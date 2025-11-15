import pandas as pd

# Leer CSV indicando que el separador es ;
df = pd.read_csv('Dades\\train.csv', sep=';')

df_sin_una = df.loc[:, df.columns != df.columns[8]]
# Mostrar nombres de columnas correctamente
#print(df.columns)

# Seleccionar las dos primeras columnas
col = pd.read_csv('Dades\\train.csv', sep=';', usecols=[df.columns[0], df.columns[27], df.columns[-1]])

families = []

for n in df_sin_una.columns[2:]:
    families.append(df_sin_una[n].unique())

print(families)