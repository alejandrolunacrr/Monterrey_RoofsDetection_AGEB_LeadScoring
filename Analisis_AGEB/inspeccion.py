import geopandas as gpd

# Tu archivo
ruta_agebs = r"C:\Users\aleja\Desktop\google_earths\Monterrey_SHP\MONTERREY.shp"

print("--- INSPECCIONANDO ARCHIVO ---")
gdf = gpd.read_file(ruta_agebs)

# 1. Ver todas las columnas disponibles
print("\n1. NOMBRES DE LAS COLUMNAS ENCONTRADAS:")
print(gdf.columns.tolist())

# 2. Ver una muestra de las primeras 5 filas
print("\n2. PRIMERAS 5 FILAS DE DATOS:")
print(gdf.head())

# 3. Intentar adivinar cuál es la columna de NSE y ver sus valores únicos
print("\n3. VALORES ÚNICOS EN COLUMNAS SOSPECHOSAS:")
possible_cols = [c for c in gdf.columns if len(gdf[c].unique()) < 20] # Columnas con pocas variantes

for col in possible_cols:
    if col != 'geometry':
        print(f"\nColumna: '{col}'")
        print(f"Valores encontrados: {gdf[col].unique()}")