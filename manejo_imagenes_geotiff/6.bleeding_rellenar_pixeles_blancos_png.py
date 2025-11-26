from osgeo import gdal
from pathlib import Path
import random
from tqdm import tqdm
import os

# --- Configuración ---
# 1. Carpeta con tus PNGs originales
PNG_FOLDER = Path(r"C:\Users\aleja\Desktop\google_earths\teselas_descargadas")

# 2. Cuántos archivos aleatorios quieres verificar
SAMPLE_SIZE = 5000
# ---------------------

# Desactivamos los logs de error de GDAL para no inundar la consola
# (podemos comentarlo si queremos ver errores)
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

print(f"Buscando PNGs en: {PNG_FOLDER}")
all_png_files = list(PNG_FOLDER.glob("*.png"))
total_files = len(all_png_files)

if total_files == 0:
    print(f"🚨 Error: No se encontraron archivos .png en {PNG_FOLDER}")
    exit()

print(f"Archivos .png encontrados: {total_files}")

# --- Ajustar el tamaño de la muestra ---
# Si hay menos archivos que el tamaño de la muestra, los analizamos todos
if total_files < SAMPLE_SIZE:
    print(f"  Aviso: Se encontraron menos de {SAMPLE_SIZE} archivos. Analizando todos ({total_files}).")
    sample_files = all_png_files
    sample_size_actual = total_files
else:
    print(f"  Seleccionando {SAMPLE_SIZE} archivos al azar...")
    # Seleccionamos una muestra aleatoria de la lista de archivos
    sample_files = random.sample(all_png_files, SAMPLE_SIZE)
    sample_size_actual = SAMPLE_SIZE

# --- Contadores ---
# Usamos un diccionario para contar cuántos archivos tiene cada número de bandas
# Ejemplo: {3: 4500, 4: 500}
band_counts = {}
error_count = 0

print(f"Iniciando análisis de {sample_size_actual} archivos...")

# --- Bucle con tqdm ---
for file_path in tqdm(sample_files, desc="Verificando bandas", unit="imagen", ncols=100):
    try:
        ds = gdal.Open(str(file_path))
        
        if ds is None:
            error_count += 1
            tqdm.write(f"Error al abrir: {file_path.name}")
            continue

        # Obtenemos el número de bandas
        count = ds.RasterCount
        
        # Actualizamos el contador para ese número de bandas
        # .get(count, 0) significa: "dame el valor de 'count', o 0 si no existe"
        band_counts[count] = band_counts.get(count, 0) + 1
        
        ds = None # Cerramos el archivo

    except Exception as e:
        error_count += 1
        # Imprimimos el error sin romper la barra
        tqdm.write(f"Error inesperado con {file_path.name}: {e}")

# Reactivamos el manejador de errores estándar
gdal.PopErrorHandler()

# --- 5. Mostrar resultados ---
print("\n--- Resultados del Análisis de Bandas ---")
print(f"Total de archivos analizados: {sample_size_actual}")

# Ordenamos por número de bandas (ej. 3, 4) para un informe claro
for bands in sorted(band_counts.keys()):
    count = band_counts[bands]
    percentage = (count / sample_size_actual) * 100
    print(f"  -> {count} archivos ({percentage:.1f}%) tienen {bands} bandas.")

if error_count > 0:
    print(f"\nErrores al abrir o leer: {error_count} archivos.")

print("-----------------------------------------")