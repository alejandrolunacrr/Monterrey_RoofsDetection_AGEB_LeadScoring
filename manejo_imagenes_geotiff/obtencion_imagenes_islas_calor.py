import ee
import sys

# --- 1. INICIALIZACIÓN ---
try:
    ee.Initialize(project='arched-alpha-475400-b7')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='arched-alpha-475400-b7')

# --- 2. REGIÓN ---
monterrey_area = ee.Geometry.Polygon(
    [[[-100.539348, 25.895389],
      [-100.539348, 25.333417],
      [-100.040251, 25.333417],
      [-100.040251, 25.895389]]])

# --- 3. FUNCIONES ---
def mask_clouds(image):
    # Usamos la banda QA para borrar nubes y sombras
    qa = image.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0)))
    return image.updateMask(mask)

def add_celsius(image):
    # CALCULO OFICIAL USGS PARA LANDSAT 8/9 LEVEL 2
    # Fórmula: (DN * 0.00341802 + 149.0) - 273.15 = Grados Celsius
    
    # Seleccionamos la banda térmica (ST_B10)
    thermal = image.select('ST_B10')
    
    # Aplicamos la fórmula lineal y convertimos a Celsius
    celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15).rename('TEMP_CELSIUS')
    
    return image.addBands(celsius)

# --- 4. COLECCIÓN (SOLO VERANO/CALOR o AÑO COMPLETO) ---
# ESTRATEGIA: Para medir "Isla de Calor", lo mejor es tomar los meses cálidos
# de todo el año. Si tomas invierno, "diluyes" el efecto del calor extremo.
# Aquí filtramos de Abril a Octubre de 2024 y 2025.

dataset = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(monterrey_area)
    .filterDate('2024-04-01', '2025-10-31') # Rango amplio
    .filter(ee.Filter.calendarRange(4, 10, 'month')) # SOLO meses de calor (Abril-Octubre)
    .filter(ee.Filter.lt('CLOUD_COVER', 40)) 
    .map(mask_clouds)
    .map(add_celsius))

# --- 5. VERIFICACIÓN ---
cantidad = dataset.size().getInfo()
print(f"Imágenes térmicas encontradas (Meses cálidos): {cantidad}")

if cantidad == 0:
    sys.exit("No hay imágenes. Ajusta filtros.")

# --- 6. COMPOSICIÓN (MEDIANA TÉRMICA) ---
# Obtenemos la mediana de temperatura en grados Celsius
imagen_final = dataset.select('TEMP_CELSIUS').median().clip(monterrey_area)

# --- 7. EXPORTACIÓN DE DATOS (GRAYSCALE) ---
# Este archivo tendrá VALORES REALES (ej. 35.5, 42.1)
task = ee.batch.Export.image.toDrive(**{
    'image': imagen_final,
    'description': 'Monterrey_LST_Celsius_Datos',
    'folder': 'GEE_Exports',
    'fileNamePrefix': 'MTY_LST_CELSIUS_2025',
    'scale': 30, # Landsat térmica es 100m resampleada a 30m
    'region': monterrey_area,
    'fileFormat': 'GeoTIFF',
    'maxPixels': 1e9
})

task.start()
print("-> Exportación iniciada: MTY_LST_CELSIUS_2025.tif")