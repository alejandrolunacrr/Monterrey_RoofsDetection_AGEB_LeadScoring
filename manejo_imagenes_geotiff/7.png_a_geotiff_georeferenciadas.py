import mercantile
from osgeo import gdal
from pathlib import Path
from tqdm import tqdm
import os

# --- Configuración ---
PNG_FOLDER = Path(r"C:\Users\aleja\Desktop\google_earths\teselas_descargadas")
GEOTIFF_FOLDER = Path("teselas_geotiff") # Asegúrate de que esta carpeta esté vacía
# ---------------------

GEOTIFF_FOLDER.mkdir(exist_ok=True)

# 1. Obtenemos la lista de .png
try:
    files = list(PNG_FOLDER.glob("*.png"))
    if not files:
        print(f"🚨 Error: No se encontraron archivos .png en ./{PNG_FOLDER}/")
        exit()
except FileNotFoundError:
    print(f"🚨 Error: La carpeta ./{PNG_FOLDER}/ no existe.")
    exit()

print(f"Encontradas {len(files)} teselas .png. Creando GeoTIFFs (¡Versión Corregida!)...")

# 2. Iteramos sobre cada .png
for png_file in tqdm(files, unit="tif"):
    try:
        z_str, x_str, y_str = png_file.stem.split('_')
        z = int(z_str)
        x = int(x_str)
        y = int(y_str)

        # 3. Obtenemos coordenadas
        bounds = mercantile.bounds(x, y, z) 
        # bounds es un objeto con: .west, .south, .east, .north
        
        geotiff_path = GEOTIFF_FOLDER / f"{png_file.stem}.tif"

        # 5. Usamos GDAL para crear el GeoTIFF
        # gdal.Translate espera las coordenadas en el orden:
        # [Oeste, Norte, Este, Sur]
        gdal.Translate(
            str(geotiff_path),
            str(png_file),
            format="GTiff",
            outputSRS="EPSG:4326",
            # --- LA CORRECCIÓN ESTÁ AQUÍ ---
            # El orden correcto es [west, north, east, south]
            outputBounds=[bounds.west, bounds.north, bounds.east, bounds.south] # <-- CAMBIO CLAVE
        )

    except Exception as e:
        print(f"\nAdvertencia: Error procesando {png_file.name}: {e}")

print(f"\n✅ {len(files)} archivos GeoTIFF (corregidos) creados en ./{GEOTIFF_FOLDER}/")