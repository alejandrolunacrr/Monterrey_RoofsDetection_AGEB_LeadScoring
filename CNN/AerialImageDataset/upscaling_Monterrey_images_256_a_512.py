import os
import glob
import rasterio
from rasterio.merge import merge
from tqdm import tqdm
from collections import defaultdict
import numpy as np

"""
========================================================================
SCRIPT DE CONSOLIDACIÓN DE TESELAS (UPSCALING LÓGICO DE 256 A 512)
========================================================================

DESCRIPCIÓN:
Este script procesa un directorio de teselas satelitales estándar descargadas
(generalmente de 256x256 píxeles, Zoom 19) y las agrupa para formar teselas
de mayor resolución (512x512 píxeles).

FUNCIONALIDAD:
1. Recorre la carpeta INPUT_DIR e identifica los archivos GeoTIFF.
2. Utiliza la nomenclatura "Z_X_Y.tif" para calcular matemáticamente qué
   teselas son adyacentes y pertenecen al mismo "padre" en un nivel de zoom
   inferior (Zoom 18).
3. Agrupa las teselas en bloques de 2x2 (4 imágenes de 256x256).
4. Realiza un mosaico georreferenciado en memoria utilizando Rasterio.merge.
5. Exporta la nueva imagen de 512x512 píxeles conservando la proyección
   geográfica correcta en OUTPUT_DIR.

OBJETIVO:
Preparar el dataset para alimentar redes neuronales que requieren entradas
de 512x512 (como U-Net con ResNet34), maximizando el contexto espacial
sin perder la resolución original por píxel.
========================================================================
"""

# --- 1. CONFIGURACIÓN ---

class CONFIG:
    # La carpeta con todas tus teselas de 256x256
    INPUT_DIR = r"C:\Users\aleja\Desktop\google_earths\teselas_geotiff4"
    
    # La nueva carpeta donde se guardarán los tiles de 512x512
    OUTPUT_DIR = r"C:\Users\aleja\Desktop\google_earths\teselas_geotiff4_512"
    
    # El zoom de tus tiles de entrada (basado en '19_...')
    INPUT_ZOOM = 19

# --------------------------

def group_and_merge_tiles(input_dir, output_dir, input_zoom):
    """
    Agrupa los tiles de entrada (asumidos 256x256) en "padres"
    y los combina en nuevos tiles (asumidos 512x512).
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Usamos defaultdict para crear listas automáticamente
    tile_groups = defaultdict(list)
    
    search_criteria = os.path.join(input_dir, "*.tif")
    input_files = glob.glob(search_criteria)
    
    if not input_files:
        print(f"¡Error! No se encontraron archivos .tif en {input_dir}")
        return

    print(f"Fase 1: Agrupando {len(input_files)} tiles...")
    
    # --- FASE 1: Agrupar tiles por su "padre" ---
    for fpath in tqdm(input_files, desc="Agrupando tiles"):
        fname = os.path.basename(fpath)
        
        # Asumimos formato ZOOM_X_Y.tif, ej: "19_115730_223363.tif"
        parts = fname.replace('.tif', '').split('_')
        
        if len(parts) != 3:
            print(f"Omitiendo archivo con nombre inesperado: {fname}")
            continue
            
        try:
            z, x, y = map(int, parts)
        except ValueError:
            print(f"Omitiendo archivo (no se pudieron leer X,Y): {fname}")
            continue
            
        if z != input_zoom:
             print(f"Omitiendo archivo (zoom inesperado): {fname}")
             continue

        # Calcular el tile "padre" (el tile de 512x512)
        # Esto es equivalente a un tile de zoom 18
        parent_z = z - 1
        parent_x = x // 2
        parent_y = y // 2
        
        # Creamos una clave única para este grupo
        parent_key = f"{parent_z}_{parent_x}_{parent_y}"
        
        # Agregamos este archivo a la lista de su padre
        tile_groups[parent_key].append(fpath)

    print(f"Agrupación completa. Se crearán {len(tile_groups)} tiles de 512x512.")

    # --- FASE 2: Combinar cada grupo ---
    print("Fase 2: Creando nuevos tiles...")
    for parent_name, tile_paths in tqdm(tile_groups.items(), desc="Combinando tiles"):
        
        src_files_to_mosaic = []
        try:
            # Abrir todos los tiles (usualmente 4) que pertenecen a este grupo
            for fp in tile_paths:
                src_files_to_mosaic.append(rasterio.open(fp))
            
            # Combinarlos en memoria
            mosaic, out_trans = merge(src_files_to_mosaic)
            
            # Copiar metadatos del primer tile y actualizarlos
            meta = src_files_to_mosaic[0].meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1], # Alto del mosaico
                "width": mosaic.shape[2],  # Ancho del mosaico
                "transform": out_trans,    # Nuevo transform geográfico
                "compress": "lzw"
            })
            
            # Definir la ruta de salida para este nuevo tile
            out_path = os.path.join(output_dir, f"{parent_name}.tif")
            
            # Escribir el nuevo tile 512x512
            with rasterio.open(out_path, "w", **meta) as dest:
                dest.write(mosaic)
                
        except Exception as e:
            print(f"Error al procesar el grupo {parent_name}: {e}")
        finally:
            # Asegurarse de cerrar todos los archivos
            for src in src_files_to_mosaic:
                src.close()

    print("--- ¡Proceso completado! ---")
    print(f"Se crearon {len(tile_groups)} tiles de 512x512 en: {output_dir}")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    group_and_merge_tiles(CONFIG.INPUT_DIR, CONFIG.OUTPUT_DIR, CONFIG.INPUT_ZOOM)