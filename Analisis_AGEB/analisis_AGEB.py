import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import os
from shapely.geometry import box
import numpy as np

"""
RESUMEN DEL SCRIPT: ANÁLISIS GEOSPACIAL DE POTENCIAL SOLAR

Este flujo de trabajo integra datos vectoriales (AGEBs) con rasters de detección de techos (salida de modelo CNN) para calificar zonas de venta en Monterrey.

FUNCIONALIDADES PRINCIPALES:
1. Ingesta de Datos: Carga polígonos de AGEBs y múltiples máscaras binarias de techos (TIFs).
2. Calibración Física: Calcula el área real por pixel proyectando a coordenadas UTM (Zona 14N) para obtener metros cuadrados precisos.
3. Extracción Zonal: Suma la superficie de techos detectados dentro de cada AGEB utilizando 'zonal_stats'.
4. Lógica de Negocio: Aplica pesos específicos según el Nivel Socioeconómico (NSE) de la zona (ej. A/B = 1.0, D = 0.0).
5. Generación de Scores:
   - Score Solar (Volumen): Total de m² útiles ponderados por NSE. Ideal para marketing masivo.
   - Score Eficiencia (Densidad): % de ocupación de techos de valor por m² de tierra. Ideal para optimizar rutas de venta directa.
6. Exportación: Genera un GeoJSON para visualización GIS y un CSV ordenado por ranking de oportunidad.
"""

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

# Ruta del Shapefile (AMAI)
ruta_agebs = r"C:\Users\aleja\Desktop\google_earths\Monterrey_SHP\MONTERREY.shp"

# Tus 3 Imágenes Satelitales (Máscaras)
lista_rasters = [
    r"C:\Users\aleja\Desktop\google_earths\CNN\MONTERREY_MASCARA_COMPLETA_tunned_512_2.tif",
    r"C:\Users\aleja\Desktop\google_earths\CNN\MONTERREY_MASCARA_COMPLETA_tunned_512_3.tif",
    r"C:\Users\aleja\Desktop\google_earths\CNN\MONTERREY_MASCARA_COMPLETA_tunned_512_4.tif"
]

# Archivos de Salida
ruta_salida_geo = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_Analisis_Solar_Completo.geojson"
ruta_salida_csv = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_Ranking_Ventas.csv"

# ==========================================
# 2. DEFINICIÓN DE FACTORES DE NEGOCIO
# ==========================================

# Columna exacta encontrada en tu archivo:
nombre_columna_nse = 'NIVEL'

# Mapeo exacto de valores encontrados -> Factor de Negocio
pesos_nse = {
    'A/B': 1.00,  # Cliente Ideal
    'C+':  0.90,  # Excelente Mercado (Crédito)
    'C':   0.60,  # Mercado Medio
    'C-':  0.25,  # Riesgo Alto
    'D+':  0.10,
    'D':   0.00,
    'E':   0.00,
    'N/D': 0.00
}

# ==========================================
# 3. CALIBRACIÓN (METROS REALES)
# ==========================================
print("--- 1. Iniciando Procesamiento ---")

if not os.path.exists(ruta_agebs):
    raise FileNotFoundError(f"No encuentro: {ruta_agebs}")

gdf_ageb = gpd.read_file(ruta_agebs)
print(f" > AGEBs cargados: {len(gdf_ageb)}")

# Inicializamos acumulador
gdf_ageb['suma_pixeles_total'] = 0.0

print("--- 2. Calibrando área de pixel (Grados a Metros) ---")
# Usamos el primer raster para calcular tamaño real
if os.path.exists(lista_rasters[0]):
    with rasterio.open(lista_rasters[0]) as src:
        raster_crs = src.crs
        res_x, res_y = src.res
        
        # Truco para obtener metros reales en Monterrey (Zona 14N)
        centro_x = src.bounds.left + (src.bounds.right - src.bounds.left) / 2
        centro_y = src.bounds.top + (src.bounds.bottom - src.bounds.top) / 2
        pixel_geom = box(centro_x, centro_y, centro_x + res_x, centro_y + res_y)
        
        gs_pixel = gpd.GeoSeries([pixel_geom], crs=raster_crs)
        gs_pixel_metros = gs_pixel.to_crs("EPSG:32614")
        area_pixel_m2 = gs_pixel_metros.area[0]
        
        print(f" > Área REAL por pixel: {area_pixel_m2:.4f} m2")
else:
    raise FileNotFoundError("No encuentro el primer raster para calibrar.")

# Alinear proyección para contar
if gdf_ageb.crs != raster_crs:
    print(" > Reproyectando AGEBs para coincidir con satélite...")
    gdf_ageb = gdf_ageb.to_crs(raster_crs)

# ==========================================
# 4. EXTRACCIÓN DE TECHOS (MULTI-RASTER)
# ==========================================
print("--- 3. Contando techos en las imágenes ---")

for i, ruta_tif in enumerate(lista_rasters):
    nombre = os.path.basename(ruta_tif)
    if not os.path.exists(ruta_tif):
        print(f" [!] Falta archivo: {nombre}")
        continue
        
    print(f" > Procesando {i+1}/{len(lista_rasters)}: {nombre}")
    try:
        stats = zonal_stats(gdf_ageb, ruta_tif, stats=["sum"], nodata=-9999)
        df_temp = pd.DataFrame(stats)
        df_temp['sum'] = df_temp['sum'].fillna(0)
        gdf_ageb['suma_pixeles_total'] += df_temp['sum'].values
    except Exception as e:
        print(f" [Error] en {nombre}: {e}")

# ==========================================
# 5. CÁLCULOS FÍSICOS
# ==========================================
print("--- 4. Calculando Métricas Físicas ---")

# Convertir pixeles a M2
gdf_ageb['area_techos_m2'] = gdf_ageb['suma_pixeles_total'] * area_pixel_m2

# Calcular Área Total del Terreno del AGEB (Reproyectando a UTM)
gdf_utm = gdf_ageb.to_crs("EPSG:32614")
gdf_ageb['area_tierra_m2'] = gdf_utm.geometry.area

# Densidad Real (Qué % del suelo está ocupado por techos detectados)
gdf_ageb['densidad_techos'] = gdf_ageb['area_techos_m2'] / gdf_ageb['area_tierra_m2']

# Limpieza
gdf_ageb = gdf_ageb.drop(columns=['suma_pixeles_total'])

# ==========================================
# 6. SCORING DE NEGOCIO (Volumen y Eficiencia)
# ==========================================
print("--- 5. Generando Scores de Venta ---")

if nombre_columna_nse in gdf_ageb.columns:
    print(f" > Cruzando con columna: {nombre_columna_nse}")
    
    def obtener_peso(valor):
        if pd.isna(valor): return 0.0
        v = str(valor).strip() 
        return pesos_nse.get(v, 0.0)

    gdf_ageb['factor_nse'] = gdf_ageb[nombre_columna_nse].apply(obtener_peso)
    
    # SCORE 1: VOLUMEN TOTAL (Para Marketing Masivo)
    # Premia zonas grandes con muchos techos.
    gdf_ageb['score_solar'] = gdf_ageb['area_techos_m2'] * gdf_ageb['factor_nse']

    # SCORE 2: EFICIENCIA (Para Cambaceo/Venta Directa)
    # Premia zonas densas donde cada puerta cuenta, sin importar el tamaño de la zona.
    # Multiplicamos por 1000 para tener una escala legible similar a la otra.
    gdf_ageb['score_eficiencia'] = gdf_ageb['densidad_techos'] * gdf_ageb['factor_nse'] * 1000

else:
    print(f" [ERROR] No encontré la columna {nombre_columna_nse}")
    gdf_ageb['score_solar'] = 0
    gdf_ageb['score_eficiencia'] = 0

# Ordenamos por Eficiencia (mejores barrios compactos primero)
gdf_ageb = gdf_ageb.sort_values(by='score_eficiencia', ascending=False)

# ==========================================
# 7. GUARDAR RESULTADOS
# ==========================================
print("--- 6. Guardando Archivos ---")

gdf_ageb.to_file(ruta_salida_geo, driver='GeoJSON')

cols_export = [
    'CVE_AGEBC', 'NOM_MUN', 'NOM_LOC', nombre_columna_nse, 
    'area_techos_m2', 'factor_nse', 
    'score_solar', 'score_eficiencia'
]
# Filtrar columnas existentes
cols_finales = [c for c in cols_export if c in gdf_ageb.columns]

gdf_ageb[cols_finales].to_csv(ruta_salida_csv, index=False)

print("-" * 50)
print("¡PROCESO COMPLETADO!")
print(f"1. Mapa (GeoJSON): {ruta_salida_geo}")
print(f"2. Tabla (CSV):    {ruta_salida_csv}")
print("-" * 50)
print("TOP 5 ZONAS POR EFICIENCIA (Alta densidad de valor):")
print(gdf_ageb[cols_finales].head(5))