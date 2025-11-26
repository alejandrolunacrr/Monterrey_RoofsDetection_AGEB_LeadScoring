# 1. Importa las librerías necesarias
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds
import mercantile
import sys # Para salir

# --- Configuración ---
RUTA_CARPETA = r"C:\Users\aleja\Desktop\google_earths\manejo_imagenes_geotiff\seleccion\seleccion4"
ZOOM_LEVEL = 19
CRS_DESTINO = "EPSG:4326"
# ---------------------

print(f"📁 Analizando archivos en: {RUTA_CARPETA}")

p = Path(RUTA_CARPETA)

# Leemos los archivos
archivos_tif = []
try:
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']:
            archivos_tif.append(f)
except FileNotFoundError:
    print(f"🚨 ¡Error Fatal! La RUTA_CARPETA no existe: {RUTA_CARPETA}")
    sys.exit()

if not archivos_tif:
    print(f"🚨 ¡Error! No se encontraron archivos .tif o .tiff en la carpeta.")
    sys.exit()

print(f"Encontrados {len(archivos_tif)} archivos. Calculando teselas individuales...")

# --- EL GRAN CAMBIO ESTÁ AQUÍ ---
# Usamos un 'set' para guardar la lista de teselas.
# Esto evita duplicados automáticamente si dos .tif comparten una tesela.
lista_maestra_teselas = set()

# Iteramos sobre cada archivo .tif individualmente
for tif_file in archivos_tif:
    try:
        with rasterio.open(tif_file) as src:
            bounds = src.bounds
            crs_origen = src.crs

            # Reproyectamos los límites de ESTE archivo
            if crs_origen.to_string() != CRS_DESTINO:
                lon1, lat1, lon2, lat2 = transform_bounds(
                    crs_origen, CRS_DESTINO, *bounds
                )
            else:
                lon1, lat1, lon2, lat2 = bounds.left, bounds.bottom, bounds.right, bounds.top
            
            # Calculamos las teselas SÓLO PARA ESTE archivo
            teselas_para_este_archivo = mercantile.tiles(
                lon1, lat1, lon2, lat2, zooms=ZOOM_LEVEL
            )
            
            # Añadimos esas teselas a nuestra lista maestra
            lista_maestra_teselas.update(teselas_para_este_archivo)

    except Exception as e:
        print(f"Advertencia: No se pudo leer {tif_file.name}. Saltando. (Error: {e})")

# --- FIN DEL CAMBIO ---

print("\n--- ¡Proceso Completado! ---")
print(f"Total de teselas ÚNICAS a descargar: {len(lista_maestra_teselas)}")

# Guardamos la lista (que ahora sí es la correcta)
with open("lista_de_teselas.txt", "w") as f:
    for tesela in lista_maestra_teselas:
        f.write(f"{tesela.x},{tesela.y},{tesela.z}\n")

print(f"\n✅ Lista de {len(lista_maestra_teselas)} teselas guardada en 'lista_de_teselas.txt'")
print("Este archivo es tu entrada para el Paso 2 (Descarga).")