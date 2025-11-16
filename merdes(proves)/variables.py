import pandas as pd

# Leer CSV indicando que el separador es ;
df = pd.read_csv('dades\\dades.csv', sep=';')

families = []

for n in df.columns[2:]:
    families.append(df[n].unique())


f = pd.DataFrame(families)
print(f)
f.to_csv('dades\\families.csv', sep=';', index=False)