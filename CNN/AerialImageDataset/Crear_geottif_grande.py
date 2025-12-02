import os
import glob
import rasterio
from rasterio.merge import merge
from tqdm import tqdm

"""
SCRIPT DE GENERACIÓN DE MOSAICOS (RASTER MERGE)

Este script consolida miles de archivos GeoTIFF individuales (tiles) en un único archivo raster continuo y georreferenciado. Es el paso final del pipeline de procesamiento para visualizar los resultados en GIS.

FUNCIONALIDADES:
1. Agregación Masiva: Busca recursivamente archivos .tif en directorios específicos y los carga en memoria.
2. Fusión Espacial: Utiliza 'rasterio.merge' para unir los tiles basándose en sus coordenadas geográficas, recalculando las dimensiones totales y la transformación afín.
3. Soporte BigTIFF: Habilita explícitamente el driver `BIGTIFF="YES"` para permitir la escritura de archivos que exceden el límite de 4GB estándar de TIFF.
4. Compresión: Aplica compresión LZW para optimizar el almacenamiento sin pérdida de calidad.

SALIDA: Genera dos mosaicos completos (Imagen Original y Máscara de Predicción) listos para análisis en QGIS/ArcGIS.
"""

def create_mosaic(input_folder, output_path):
    """
    Combina todos los archivos .tif de una carpeta en un solo GeoTIFF (mosaico).
    """
    print(f"Iniciando mosaico para: {input_folder}")
    
    # 1. Encontrar todos los archivos .tif en la carpeta
    search_criteria = os.path.join(input_folder, "*.tif")
    tif_files = glob.glob(search_criteria)
    
    if not tif_files:
        print(f"¡Error! No se encontraron archivos .tif en {input_folder}")
        return

    print(f"Se encontraron {len(tif_files)} tiles para combinar.")

    # 2. Abrir todos los archivos
    src_files_to_mosaic = []
    for fp in tqdm(tif_files, desc="Abriendo archivos"):
        try:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)
        except Exception as e:
            print(f"No se pudo abrir {fp}: {e}")
            
    if not src_files_to_mosaic:
        print("No se pudieron abrir los archivos. Abortando.")
        return

    # 3. Combinar los archivos
    print("Combinando archivos en memoria...")
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # 4. Copiar los metadatos y actualizarlos
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw",
        "BIGTIFF": "YES"  # <-- ¡ESTA ES LA CORRECCIÓN!
    })

    # 5. Escribir el mosaico final a un nuevo archivo .tif
    print(f"Guardando mosaico en: {output_path}")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
        
    # 6. Cerrar todos los archivos abiertos
    print("Cerrando archivos...")
    for src in src_files_to_mosaic:
        src.close()
        
    print("¡Mosaico completado!")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    
    # --- PROCESO 1: Crear el mosaico de MÁSCARAS ---
    # (Tu log mostró que esta carpeta no se encontró, asegúrate que la ruta sea correcta)
    INPUT_MASK_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_predicciones_mascaras\predicciones_monterrey_SOLIDOS\cuadrante4"
    OUTPUT_MOSAIC_MASK = r"C:\Users\aleja\Desktop\google_earths\CNN\Imagenes y mascaras completas de MTY\MONTERREY_MASCARA_NOM_4.tif"
    create_mosaic(INPUT_MASK_DIR, OUTPUT_MOSAIC_MASK)

    # --- PROCESO 2: Crear el mosaico de IMÁGENES ORIGINALES ---
    INPUT_IMAGE_DIR = r"C:\Users\aleja\Desktop\google_earths\teselas_geotiff4_512kk" 
    OUTPUT_MOSAIC_IMAGE = r"C:\Users\aleja\Desktop\google_earths\CNN\Imagenes y mascaras completas de MTY\MONTERREY_IMAGEN_COMPLETA_4.tif"
    create_mosaic(INPUT_IMAGE_DIR, OUTPUT_MOSAIC_IMAGE)