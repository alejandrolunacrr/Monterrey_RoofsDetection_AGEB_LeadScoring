import geopandas as gpd
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURACIÓN Y CONSTANTES
# ==========================================
ruta_entrada = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_Analisis_Final.geojson"
ruta_salida_geo = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_Mapa_Dimensionado_Detallado.geojson"
ruta_salida_reporte_ageb = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_Reporte_Financiero_Por_AGEB.csv"
ruta_salida_global = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_Evolucion_Global_Anual.csv"
ruta_imagen_graficas = r"C:\Users\aleja\Desktop\google_earths\Analisis_AGEB\Monterrey_Graficas_Globales.png"

# Factores Técnicos
IRRADIACION = 1950.0
PR = 0.75
GCR_TECNICO = 0.50 
FACTOR_CO2 = 0.423 # kg CO2 / kWh

# Factores Económicos
INFLACION_LUZ = 0.05
DEGRADACION_PANEL = 0.005
PRECIOS = {'A/B': 6.20, 'C+': 3.80, 'Resto': 1.20}

# Curva de Adopción (Simulación de penetración de mercado)
PENETRACION_INICIAL = 0.02 # 2% en 2026
PENETRACION_FINAL   = 0.35 # 35% en 2040

# Factores de Uso de Techo por Nivel Socioeconómico (NSE)
FACTOR_USO_NSE = {
    'A/B': 1.0,   
    'C+':  0.7,  
    'C':   0.60,  
    'C-':  0.40,  
    'D+':  0.20,  
    'D':   0.15,  
    'E':   0.15,  
    'N/D': 0.10,  
}

# ==========================================
# 2. CÁLCULO BASE POR AGEB
# ==========================================
print("--- 1. Cargando datos y Calculando Potencial Base ---")

if not os.path.exists(ruta_entrada): 
    raise FileNotFoundError(f"No se encontró el archivo: {ruta_entrada}")

gdf = gpd.read_file(ruta_entrada)

# Limpieza básica
gdf['VIVIENDAS_INEGI'] = pd.to_numeric(gdf['VIVIENDAS_INEGI'], errors='coerce').fillna(0)
gdf['m2_techo_total'] = pd.to_numeric(gdf['m2_techo_total'], errors='coerce').fillna(0)

# Funciones auxiliares
def get_factor_uso(nivel): 
    return FACTOR_USO_NSE.get(str(nivel).strip(), 0.3)

def get_precio_luz(nivel): 
    v = str(nivel).strip()
    if v in ['A/B']: return PRECIOS['A/B']
    if v in ['C+', 'B']: return PRECIOS['C+'] 
    return PRECIOS['Resto']

gdf['factor_uso'] = gdf['NIVEL'].apply(get_factor_uso)
gdf['tarifa_base'] = gdf['NIVEL'].apply(get_precio_luz)

# Potencial Técnico Total
gdf['area_panel_ajustada'] = gdf['m2_techo_total'] * GCR_TECNICO * gdf['factor_uso']

# Generación base anual (kWh) al 100% de saturación
gdf['gen_kwh_potencial_100'] = np.where(
    gdf['VIVIENDAS_INEGI'] > 0,
    gdf['area_panel_ajustada'] * 0.205 * IRRADIACION * PR,
    0.0
)

# ==========================================
# 3. PROYECCIÓN ANUAL (2026-2040)
# ==========================================
print("--- 2. Proyectando Escenarios (2026-2040) ---")

gdf['Acumulado_15y_Dinero_MXN'] = 0.0
gdf['Acumulado_15y_Energia_GWh'] = 0.0
gdf['Acumulado_15y_CO2_Ton'] = 0.0

anos_clave = [2026, 2030, 2035, 2040]
anios = list(range(2026, 2041))
resultados_globales = []

for i, anio in enumerate(anios):
    # Factores dinámicos del año
    progreso = i / (len(anios) - 1)
    tasa_adopcion = PENETRACION_INICIAL + (progreso * (PENETRACION_FINAL - PENETRACION_INICIAL))
    factor_degradacion = (1 - DEGRADACION_PANEL) ** i
    factor_inflacion = (1 + INFLACION_LUZ) ** i
    
    # Cálculos vectorizados
    generacion_anio_kwh = (gdf['gen_kwh_potencial_100'] * tasa_adopcion * factor_degradacion)
    ahorro_anio_mxn = generacion_anio_kwh * (gdf['tarifa_base'] * factor_inflacion)
    co2_anio_ton = (generacion_anio_kwh * FACTOR_CO2) / 1000
    
    # Guardar columnas de años clave
    if anio in anos_clave:
        gdf[f'Gen_{anio}_MWh'] = generacion_anio_kwh / 1000 
        gdf[f'Ahorro_{anio}_MXN'] = ahorro_anio_mxn
        gdf[f'CO2_{anio}_Ton'] = co2_anio_ton
        
    # Acumulados totales
    gdf['Acumulado_15y_Dinero_MXN'] += ahorro_anio_mxn
    gdf['Acumulado_15y_Energia_GWh'] += (generacion_anio_kwh / 1e6)
    gdf['Acumulado_15y_CO2_Ton'] += co2_anio_ton

    # Datos para CSV y Gráficas Globales
    resultados_globales.append({
        'Año': anio,
        'Tasa_Adopcion_%': round(tasa_adopcion * 100, 2),
        'Generacion_Total_GWh': generacion_anio_kwh.sum() / 1e6,
        'Ahorro_Total_MDP': ahorro_anio_mxn.sum() / 1e6,
        'CO2_Total_Ton': co2_anio_ton.sum()
    })

# ==========================================
# 4. EXPORTACIÓN DE ARCHIVOS
# ==========================================
print("--- 3. Guardando Archivos de Datos ---")
gdf = gdf.replace([np.inf, -np.inf], 0).fillna(0)

# A. GeoJSON
try:
    if os.path.exists(ruta_salida_geo): os.remove(ruta_salida_geo)
    gdf.to_file(ruta_salida_geo, driver='GeoJSON')
except Exception as e:
    print(f" [!] Error guardando GeoJSON: {e}")

# B. DataFrame Global
df_global = pd.DataFrame(resultados_globales)
df_global.to_csv(ruta_salida_global, index=False)

# C. CSV Detallado por AGEB
cols_base = ['CVEGEO', 'NOM_LOC', 'NIVEL', 'VIVIENDAS_INEGI']
cols_acumuladas = ['Acumulado_15y_Energia_GWh', 'Acumulado_15y_Dinero_MXN', 'Acumulado_15y_CO2_Ton']
cols_anuales = [c for c in gdf.columns if any(str(y) in c for y in anos_clave)]
cols_finales = [c for c in (cols_base + cols_acumuladas + cols_anuales) if c in gdf.columns]

try:
    gdf[cols_finales].sort_values('Acumulado_15y_Dinero_MXN', ascending=False).to_csv(ruta_salida_reporte_ageb, index=False)
except Exception as e:
    print(f" [!] Error guardando CSV Detallado: {e}")

# ==========================================
# 5. GENERACIÓN DE GRÁFICAS (SOLO MONTERREY)
# ==========================================
print("--- 4. Generando Gráficas Globales ---")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Datos (eje X compartido)
x_anios = df_global['Año']

# 1. Gráfica Dinero (Verde)
ax_din = axes[0]
ax_din.plot(x_anios, df_global['Ahorro_Total_MDP'], marker='o', color='green', linewidth=2.5)
ax_din.fill_between(x_anios, df_global['Ahorro_Total_MDP'], color='green', alpha=0.1)
ax_din.set_title('Ahorro Económico Anual (MDP)', fontsize=12, fontweight='bold')
ax_din.set_ylabel('Millones de Pesos')
ax_din.grid(True, linestyle='--', alpha=0.6)

# 2. Gráfica CO2 (Gris)
ax_co2 = axes[1]
ax_co2.plot(x_anios, df_global['CO2_Total_Ton'], marker='s', color='#555555', linewidth=2.5)
ax_co2.fill_between(x_anios, df_global['CO2_Total_Ton'], color='#555555', alpha=0.1)
ax_co2.set_title('Reducción de CO2 (Toneladas)', fontsize=12, fontweight='bold')
ax_co2.set_ylabel('Toneladas CO2')
ax_co2.grid(True, linestyle='--', alpha=0.6)

# 3. Gráfica Energía (Naranja)
ax_ene = axes[2]
ax_ene.plot(x_anios, df_global['Generacion_Total_GWh'], marker='^', color='orange', linewidth=2.5)
ax_ene.fill_between(x_anios, df_global['Generacion_Total_GWh'], color='orange', alpha=0.1)
ax_ene.set_title('Generación Energía (GWh)', fontsize=12, fontweight='bold')
ax_ene.set_ylabel('GWh')
ax_ene.grid(True, linestyle='--', alpha=0.6)

# Títulos y Ajustes
plt.suptitle('Proyección Impacto Solar Monterrey (2026-2040)', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Guardar
plt.savefig(ruta_imagen_graficas, dpi=300, bbox_inches='tight')
print(f"5. Imagen de gráficas guardada en: {ruta_imagen_graficas}")

print("-" * 50)
print("¡PROCESO FINALIZADO!")
print("-" * 50)
plt.show()