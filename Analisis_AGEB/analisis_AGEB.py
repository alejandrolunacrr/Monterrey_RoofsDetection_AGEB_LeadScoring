import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import os
from shapely.geometry import box
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
# Mapa Vectorial
ruta_agebs = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_SHP\MONTERREY.shp"

# Tus 3 Imágenes de Techos (CNN)
lista_rasters_techos = [
    r"C:\Users\aleja\Desktop\google_earths\CNN\Imagenes y mascaras completas de MTY\MONTERREY_MASCARA_NOM_2.tif",
    r"C:\Users\aleja\Desktop\google_earths\CNN\Imagenes y mascaras completas de MTY\MONTERREY_MASCARA_NOM_3.tif",
    r"C:\Users\aleja\Desktop\google_earths\CNN\Imagenes y mascaras completas de MTY\MONTERREY_MASCARA_NOM_4.tif"
]

# Datos Complementarios
ruta_censo = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\RESAGEBURB_19CSV20.csv"
ruta_calor = r"C:\Users\aleja\Desktop\google_earths\CNN\Imagenes y mascaras completas de MTY\MTY_LST_CELSIUS_2025.tif"

# Archivos de Salida (Nombres definitivos)
ruta_salida_geo = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_Analisis_Final.geojson"
ruta_salida_csv = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_Ranking_Final.csv"

# Nombre de la columna de Nivel Socioeconómico en tu SHP
nombre_columna_nse = 'NIVEL'

# ==========================================
# 2. ESTRATEGIAS DE NEGOCIO (PESOS)
# ==========================================

# 1. PREMIUM (Ticket Alto): Prioridad A/B (Ballenas)
pesos_premium = {'A/B': 2.5, 'C+': 1.1, 'C': 0.5, 'C-': 0.1, 'D+': 0.0, 'D': 0.0, 'E': 0.0}

# 2. MASIVO (Volumen Comercial): Prioridad C+ y B (Cardumen)
pesos_masivo  = {'A/B': 0.9, 'C+': 1.0, 'B': 0.9, 'C': 0.9, 'C-': 0.5, 'D+': 0.1, 'D': 0.0}

# 3. SOCIAL (Subsidios/Gobierno): Prioridad D, D+, C- (Vulnerabilidad)
pesos_social  = {'A/B': 0.0, 'C+': 0.0, 'B': 0.0, 'C': 0.2, 'C-': 0.9, 'D+': 1.0, 'D': 1.0, 'E': 0.8}

# ==========================================
# 3. CARGA DE DATOS
# ==========================================
print("--- 1. Cargando Datos Base ---")

# A) Cargar Mapa AGEBs
if not os.path.exists(ruta_agebs): raise FileNotFoundError(f"Falta SHP: {ruta_agebs}")
gdf_ageb = gpd.read_file(ruta_agebs)
print(f" > AGEBs cargados: {len(gdf_ageb)}")

# B) Cargar y Limpiar Censo (Viviendas)
print(" > Procesando Censo INEGI...")
if not os.path.exists(ruta_censo): raise FileNotFoundError(f"Falta CSV: {ruta_censo}")

try:
    df_censo = pd.read_csv(ruta_censo, encoding='utf-8-sig', low_memory=False)
except:
    df_censo = pd.read_csv(ruta_censo, encoding='latin-1', low_memory=False)

# Normalizar columnas
df_censo.rename(columns={'ENTIDAD': 'ENT', 'MUNICIPIO': 'MUN', 'LOCALIDAD': 'LOC'}, inplace=True)

# Filtros (Totales por AGEB)
df_censo['MZA'] = pd.to_numeric(df_censo['MZA'], errors='coerce')
df_censo = df_censo[df_censo['MZA'] == 0].copy()

# Crear llave CVEGEO
df_censo['CVEGEO'] = (
    df_censo['ENT'].astype(str).str.zfill(2) + 
    df_censo['MUN'].astype(str).str.zfill(3) + 
    df_censo['LOC'].astype(str).str.zfill(4) + 
    df_censo['AGEB'].astype(str).str.zfill(4)
)
df_censo['VIVIENDAS_INEGI'] = pd.to_numeric(df_censo['TVIVPAR'], errors='coerce').fillna(1)

# Cruzar con el Mapa
col_llave_shp = 'CVEGEO'
if 'CVEGEO' not in gdf_ageb.columns:
    if 'CVE_AGEB' in gdf_ageb.columns: col_llave_shp = 'CVE_AGEB'
    elif 'CVE_AGEBC' in gdf_ageb.columns: col_llave_shp = 'CVE_AGEBC'

print(f" > Cruzando llaves: Mapa[{col_llave_shp}] vs Censo[CVEGEO]")
gdf_ageb = gdf_ageb.merge(df_censo[['CVEGEO', 'VIVIENDAS_INEGI']], left_on=col_llave_shp, right_on='CVEGEO', how='left')
gdf_ageb['VIVIENDAS_INEGI'] = gdf_ageb['VIVIENDAS_INEGI'].fillna(1)

# ==========================================
# 4. PROCESAMIENTO DE IMÁGENES
# ==========================================
print("--- 2. Procesando Imágenes (Techos y Calor) ---")

# A) Calibración de Área usando la primera imagen
archivo_calib = lista_rasters_techos[0]
if not os.path.exists(archivo_calib): raise FileNotFoundError(f"Falta imagen: {archivo_calib}")

with rasterio.open(archivo_calib) as src:
    raster_crs = src.crs
    res_x, res_y = src.res
    centro_x = src.bounds.left + (src.bounds.right - src.bounds.left) / 2
    centro_y = src.bounds.top + (src.bounds.bottom - src.bounds.top) / 2
    pixel_geom = box(centro_x, centro_y, centro_x + res_x, centro_y + res_y)
    gs_pixel = gpd.GeoSeries([pixel_geom], crs=raster_crs).to_crs("EPSG:32614")
    area_pixel_m2 = gs_pixel.area[0]
    print(f" > Área calibrada por pixel: {area_pixel_m2:.4f} m2")

# Alinear proyecciones
if gdf_ageb.crs != raster_crs:
    print(" > Reproyectando mapa vectorial...")
    gdf_ageb = gdf_ageb.to_crs(raster_crs)

# B) Suma Iterativa de Techos (Tus 3 archivos)
gdf_ageb['suma_pixeles_total'] = 0.0

for i, ruta_tif in enumerate(lista_rasters_techos):
    nombre = os.path.basename(ruta_tif)
    if os.path.exists(ruta_tif):
        print(f"   + Procesando imagen {i+1}/3: {nombre}")
        stats = zonal_stats(gdf_ageb, ruta_tif, stats=["sum"], nodata=0)
        valores = [x['sum'] if x['sum'] is not None else 0 for x in stats]
        gdf_ageb['suma_pixeles_total'] += valores
    else:
        print(f"   [!] ALERTA: No encontré {nombre}")

# Convertir pixeles a metros cuadrados reales
# Dividimos entre 255 porque ZonalStats sumó el valor de color (255), no la cantidad (1).
gdf_ageb['m2_techo_total'] = (gdf_ageb['suma_pixeles_total'] / 255) * area_pixel_m2

# C) Temperatura (LST)
print(" > Procesando Mapa de Calor...")
if os.path.exists(ruta_calor):
    stats_calor = zonal_stats(gdf_ageb, ruta_calor, stats=["mean"], nodata=-9999)
    gdf_ageb['temp_celsius'] = pd.DataFrame(stats_calor)['mean'].fillna(25)
else:
    print("   [!] No hay mapa de calor. Usando default 25°C.")
    gdf_ageb['temp_celsius'] = 25.0

# ==========================================
# 5. CÁLCULO DE SCORES (LOS 3 INDICADORES)
# ==========================================
print("--- 3. Calculando Triple Score ---")

# Métricas Físicas Base
gdf_ageb['m2_por_casa'] = gdf_ageb['m2_techo_total'] / gdf_ageb['VIVIENDAS_INEGI']

gdf_utm = gdf_ageb.to_crs("EPSG:32614")
gdf_ageb['area_tierra_m2'] = gdf_utm.geometry.area
gdf_ageb['densidad_zona'] = gdf_ageb['m2_techo_total'] / gdf_ageb['area_tierra_m2']

# Pendiente suave (/100) y Techo bajo (1.25)
# Significado: Por cada 10°C extra, sube solo 10% el interés.
gdf_ageb['factor_urgencia'] = 1 + ((gdf_ageb['temp_celsius'] - 20) / 100)
gdf_ageb['factor_urgencia'] = gdf_ageb['factor_urgencia'].clip(lower=0.9, upper=1.2)

# Asignar Pesos por NSE
def get_peso(valor, tabla):
    v = str(valor).strip()
    return tabla.get(v, 0.0)

if nombre_columna_nse in gdf_ageb.columns:
    gdf_ageb['peso_premium'] = gdf_ageb[nombre_columna_nse].apply(lambda x: get_peso(x, pesos_premium))
    gdf_ageb['peso_masivo']  = gdf_ageb[nombre_columna_nse].apply(lambda x: get_peso(x, pesos_masivo))
    gdf_ageb['peso_social']  = gdf_ageb[nombre_columna_nse].apply(lambda x: get_peso(x, pesos_social))
else:
    print(" [!] No encontré columna de NSE. Los pesos serán 0.")
    gdf_ageb['peso_premium'] = 0
    gdf_ageb['peso_masivo'] = 0
    gdf_ageb['peso_social'] = 0

# --- FÓRMULAS MAESTRAS DE RANKING ---

# 1. Calcular Scores Brutos (Cantidades absolutas)
gdf_ageb['SCORE_PREMIUM'] = (gdf_ageb['m2_por_casa'] * (gdf_ageb['peso_premium'] ** 2)) * gdf_ageb['factor_urgencia']
gdf_ageb['SCORE_MASIVO']  = (gdf_ageb['densidad_zona'] * gdf_ageb['peso_masivo'] * 1000) * gdf_ageb['factor_urgencia']
# Social: Calor al cuadrado para priorizar salud
gdf_ageb['SCORE_SOCIAL']  = (gdf_ageb['densidad_zona'] * gdf_ageb['peso_social'] * 1000) * (gdf_ageb['factor_urgencia'] ** 2)

# 2. NORMALIZACIÓN POR PERCENTILES (CORRECCIÓN IMPORTANTE)
# Esto asegura que los datos se distribuyan del 0 al 100 uniformemente en los 3 mapas.
# Evita que un "outlier" aplaste a los demás.

print("   > Aplicando normalización por Percentiles a los 3 indicadores...")

for tipo in ['PREMIUM', 'MASIVO', 'SOCIAL']:
    col_score = f'SCORE_{tipo}'
    col_rank = f'RANK_{tipo}_0_100'
    
    # Inicializamos la columna en 0
    gdf_ageb[col_rank] = 0.0
    
    # Solo rankeamos los AGEBs que tienen algún score positivo (para ignorar zonas muertas)
    mask = gdf_ageb[col_score] > 0
    
    if mask.any():
        # rank(pct=True) te da un valor de 0.0 a 1.0 según la posición en la tabla
        # Multiplicamos por 100 para tener escala 0-100
        gdf_ageb.loc[mask, col_rank] = gdf_ageb.loc[mask, col_score].rank(pct=True) * 100

# ==========================================
# 6. LIMPIEZA FINAL Y EXPORTACIÓN
# ==========================================
print("--- 4. Exportando Resultados ---")

# Limpieza Profunda para QGIS (Adiós errores de Infinito/NaN)
cols_limpiar = [
    'm2_por_casa', 'densidad_zona', 'm2_techo_total',
    'SCORE_PREMIUM', 'SCORE_MASIVO', 'SCORE_SOCIAL',
    'RANK_PREMIUM_0_100', 'RANK_MASIVO_0_100', 'RANK_SOCIAL_0_100'
]

for col in cols_limpiar:
    if col in gdf_ageb.columns:
        # Reemplazar infinitos con 0 y nulos con 0
        gdf_ageb[col] = gdf_ageb[col].replace([np.inf, -np.inf], 0).fillna(0)

# Exportar Mapa (GeoJSON)
print(f" > Guardando GeoJSON: {os.path.basename(ruta_salida_geo)}")
try:
    if os.path.exists(ruta_salida_geo): os.remove(ruta_salida_geo) # Borrar anterior si existe
    gdf_ageb.to_file(ruta_salida_geo, driver='GeoJSON')
except Exception as e:
    print(f" [!] Error guardando GeoJSON (quizás está abierto en QGIS): {e}")

# Exportar Tabla (CSV)
print(f" > Guardando CSV: {os.path.basename(ruta_salida_csv)}")
cols_export = [
    'CVEGEO', 'NOM_LOC', nombre_columna_nse, 
    'VIVIENDAS_INEGI', 'temp_celsius', 'm2_techo_total',
    'RANK_PREMIUM_0_100', 'RANK_MASIVO_0_100', 'RANK_SOCIAL_0_100'
]
# Seleccionar solo columnas existentes
cols_finales = [c for c in cols_export if c in gdf_ageb.columns]

# Guardar ordenado por potencial Premium
try:
    gdf_ageb[cols_finales].sort_values('RANK_PREMIUM_0_100', ascending=False).to_csv(ruta_salida_csv, index=False)
except Exception as e:
    print(f" [!] Error guardando CSV (quizás está abierto en Excel): {e}")

print("-" * 50)
print(f"¡ANÁLISIS TRIPLE COMPLETADO!")
print(f"1. Mapa listo para QGIS: {ruta_salida_geo}")
print(f"2. Tabla Excel: {ruta_salida_csv}")
print("-" * 50)