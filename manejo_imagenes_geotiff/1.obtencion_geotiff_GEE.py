import ee
import ee.mapclient

# --- INICIALIZACIÓN ---
try:
    ee.Initialize(project='arched-alpha-475400-b7')
    print("La API de Google Earth Engine se ha inicializado correctamente.")
except Exception as e:
    print("Autenticando...")
    ee.Authenticate()
    ee.Initialize(project='arched-alpha-475400-b7')

# --- DEFINIR LA REGIÓN DE INTERÉS (Bounding Box) ---
monterrey_area = ee.Geometry.Polygon(
    [[[-100.539348, 25.895389],  # Esquina Superior Izquierda (Oeste-Norte)
      [-100.539348, 25.333417],  # Esquina Inferior Izquierda (Oeste-Sur)
      [-100.040251, 25.333417],  # Esquina Inferior Derecha (Este-Sur)
      [-100.040251, 25.895389]]]) # Esquina Superior Derecha (Este-Norte)

# 2. --- SELECCIONAR IMÁGENES Y OBTENER FECHA ---
collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(monterrey_area)
              .filterDate('2025-01-01', '2025-10-19') # Rango de fechas más reciente
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1)) # Filtro de nubes más estricto
              .sort('CLOUDY_PIXEL_PERCENTAGE'))

# Verificamos si la colección tiene imágenes antes de continuar
collection_size = collection.size().getInfo()
if collection_size == 0:
    print("No se encontraron imágenes para la región y el rango de fechas especificados.")
    exit()

# Obtenemos la fecha de la imagen con menos nubes para usarla en el nombre del archivo
first_image = collection.first()
image_date_str = first_image.date().format('YYYY-MM-dd').getInfo()

print(f"La imagen base para el mosaico es del: {image_date_str}")

# Creamos el mosaico final
image = collection.mosaic().select(['B4', 'B3', 'B2', 'B8', 'B8A', 'B11', 'B12'])

# 3. --- CONFIGURACIÓN DE LA CUADRÍCULA ---
GRID_ROWS = 40
GRID_COLS = 40

# Obtenemos las coordenadas del rectángulo que encierra nuestra área de interés.
coords = monterrey_area.bounds().getInfo()['coordinates'][0]
min_lon = min(p[0] for p in coords)
min_lat = min(p[1] for p in coords)
max_lon = max(p[0] for p in coords)
max_lat = max(p[1] for p in coords)

tile_width = (max_lon - min_lon) / GRID_COLS
tile_height = (max_lat - min_lat) / GRID_ROWS

print(f"\nEl área se dividirá en una cuadrícula de {GRID_ROWS}x{GRID_COLS}, generando {GRID_ROWS * GRID_COLS} imágenes.")

# 4. --- BUCLE PARA GENERAR Y EXPORTAR CADA TESELA ---
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        tile_min_lon = min_lon + col * tile_width
        tile_max_lon = min_lon + (col + 1) * tile_width
        tile_min_lat = min_lat + row * tile_height
        tile_max_lat = min_lat + (row + 1) * tile_height

        tile_geometry = ee.Geometry.Polygon(
            [[[tile_min_lon, tile_max_lat],
              [tile_min_lon, tile_min_lat],
              [tile_max_lon, tile_min_lat],
              [tile_max_lon, tile_max_lat]]])

        # --- NOMBRE DE ARCHIVO CON FECHA ---
        file_name = f'{row}_{col}'
        
        export_params = {
            'image': image,
            'description': file_name,
            'folder': 'GEE_Exports',
            'fileNamePrefix': file_name,
            'scale': 10,
            'region': tile_geometry,
            'fileFormat': 'GeoTIFF',
            'maxPixels': 1e13
        }

        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()

        print(f"-> Tarea de exportación iniciada para la tesela ({row}, {col}): {file_name}")

print("\n¡Todas las tareas de exportación han sido iniciadas!")