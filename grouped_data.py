import pandas as pd

# Llegeix el CSV amb separador original
df = pd.read_csv("dades.csv", sep=";")

print("Columnes amb índex:")
for i, col in enumerate(df.columns):
    print(i, col)

# --- Índex de columnes ---
group_idx = [0, 1]   
sum_idx   = [30]
keep_idx  = [
    3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 31
]

# Filtrar índexs correctes
max_idx = len(df.columns) - 1
group_idx = [i for i in group_idx if i <= max_idx]
sum_idx   = [i for i in sum_idx   if i <= max_idx]
keep_idx  = [i for i in keep_idx  if i <= max_idx]

# Convertir índexs a noms
group_cols = [df.columns[i] for i in group_idx]
sum_cols   = [df.columns[i] for i in sum_idx]
keep_cols  = [df.columns[i] for i in keep_idx]

print("\nAgrupant per:", group_cols)
print("Sumant columnes:", sum_cols)
print("Mantenint columnes:", keep_cols)

# --- Agrupació ---
df_sum = df[group_cols + sum_cols].groupby(group_cols, as_index=False).sum()
df_keep = df[group_cols + keep_cols].drop_duplicates(subset=group_cols)
df_final = pd.merge(df_sum, df_keep, on=group_cols, how="left")

# Guardar en format Excel-friendly
df_final.to_csv("grouped_data.csv", index=False, sep=";")

print("\n✔️ grouped_data.csv creat i llegible per Excel!")
