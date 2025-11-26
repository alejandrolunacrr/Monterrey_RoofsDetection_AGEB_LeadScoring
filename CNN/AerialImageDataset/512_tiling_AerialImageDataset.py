import os
import rasterio
import glob  # Para buscar archivos
import shutil # Para mover archivos
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm # Para barras de progreso

"""
SCRIPT DE PRE-PROCESAMIENTO DE IMÁGENES (TILING)

Este script genera un dataset de entrenamiento y validación cortando imágenes satelitales grandes (GeoTIFF) en parches (tiles) manejables para modelos de segmentación semántica (CNN).

FUNCIONALIDADES PRINCIPALES:
1. Generación de Tiles: Aplica una ventana deslizante para cortar imágenes en parches de 512x512 píxeles con un solapamiento (overlap) de 55 píxeles para evitar bordes cortados.
2. Procesamiento de Máscaras (Ground Truth): 
   - Lee la máscara original.
   - Binariza estrictamente los valores (0 = fondo, 1 = edificio).
   - Escala los valores a 0-255 (uint8) para compatibilidad estándar con frameworks de visión por computadora.
3. Estructura de Salida: Organiza automáticamente los archivos en carpetas 'train' y 'validation' separando 'images' y 'masks'.
4. Filtrado: Incluye lógica para descartar parches vacíos o corruptos (actualmente configurado para conservar todo el dataset).

ENTRADA: Carpeta con subcarpetas 'train' y 'validation' (imágenes originales).
SALIDA: Carpeta 'traintiled512' con los recortes procesados.
"""

def create_tiles(
    image_dir, 
    mask_dir, 
    output_img_dir, 
    output_mask_dir, 
    tile_size=512, 
    overlap=55, 
    empty_threshold=0.0, 
    full_threshold=1.0
):
    """
    Corta imágenes y máscaras GeoTIFF en parches (tiles) de un tamaño fijo con solapamiento.
    Esta función ya tiene la corrección para guardar máscaras como 0-255.
    """
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    print(f"Buscando imágenes en: {image_dir}")
    
    step = tile_size - overlap
    image_files = glob.glob(os.path.join(image_dir, '*.tif'))
    
    if not image_files:
        print(f"ADVERTENCIA: No se encontraron archivos .tif en {image_dir}")
        return

    print(f"Encontrados {len(image_files)} archivos de imagen. Empezando a parchear...")

    for img_path in tqdm(image_files, desc="Procesando imágenes grandes"):
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, img_name)
        
        if not os.path.exists(mask_path):
            print(f"ADVERTENCIA: No se encontró la máscara {mask_path} para {img_name}. Omitiendo.")
            continue

        try:
            with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
                
                if src_img.width != src_mask.width or src_img.height != src_mask.height:
                    print(f"ERROR: Dimensiones no coinciden para {img_name}. Omitiendo.")
                    continue
                    
                width, height = src_img.width, src_img.height
                
                for r in range(0, height, step):
                    for c in range(0, width, step):
                        
                        window = Window(c, r, tile_size, tile_size)
                        
                        if r + tile_size > height or c + tile_size > width:
                            continue
                            
                        # --- MODIFICACIÓN 1: Leer y Binarizar Máscara (SOLO 0 y 1) ---
                        tile_mask_rgb = src_mask.read(window=window)
                        tile_mask_binary_0_1 = (tile_mask_rgb[0] > 0).astype(np.uint8)
                        
                        # --- MODIFICACIÓN 2: Filtrar usando la máscara 0-1 ---
                        building_percentage = np.mean(tile_mask_binary_0_1)
                        
                        if building_percentage < empty_threshold:
                            continue 
                        if building_percentage > full_threshold:
                            continue 

                        # --- Si el parche es bueno, lo leemos y guardamos ---
                        tile_img = src_img.read(window=window)
                        tile_transform = rasterio.windows.transform(window, src_img.transform)
                        
                        # --- Guarda el Parche de Imagen ---
                        img_meta = src_img.meta.copy()
                        img_meta.update({
                            'height': tile_size, 'width': tile_size,
                            'transform': tile_transform, 'compress': 'lzw'
                        })
                        tile_img_name = f"{os.path.splitext(img_name)[0]}_tile_r{r}_c{c}.tif"
                        tile_img_path = os.path.join(output_img_dir, tile_img_name)
                        
                        with rasterio.open(tile_img_path, 'w', **img_meta) as dst_img:
                            dst_img.write(tile_img)

                        # --- MODIFICACIÓN 3: Escalar a 255 ANTES de guardar ---
                        tile_mask_binary_0_255 = tile_mask_binary_0_1 * 255
                        
                        # --- Guarda el Parche de Máscara (ya binarizado) ---
                        mask_meta = src_mask.meta.copy()
                        mask_meta.update({
                            'height': tile_size, 'width': tile_size,
                            'transform': tile_transform, 'compress': 'lzw',
                            'count': 1, 
                            'dtype': 'uint8'
                        })
                        
                        tile_mask_name = f"{os.path.splitext(img_name)[0]}_tile_r{r}_c{c}.tif"
                        tile_mask_path = os.path.join(output_mask_dir, tile_mask_name)

                        with rasterio.open(tile_mask_path, 'w', **mask_meta) as dst_mask:
                            # Escribe la máscara 0-255
                            dst_mask.write(tile_mask_binary_0_255, 1)
        
        except Exception as e:
            print(f"ERROR procesando {img_name}: {e}")

    print(f"Procesamiento de {image_dir} completado.")

# --- CONFIGURACIÓN PRINCIPAL (SIMPLIFICADA) ---

# 1. Define tu ruta de ENTRADA (la que contiene 'train' y 'validation')
INPUT_SOURCE_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\AerialImageDataset\train" 

# 2. Define tu ruta de SALIDA (donde se guardarán los parches)
OUTPUT_TILES_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\AerialImageDataset\traintiled512"

# 3. Define los parámetros de los parches
TILE_SIZE = 512
OVERLAP = 55
EMPTY_THRESHOLD = 0.0  # (0.0 y 1.0 para no descartar nada)
FULL_THRESHOLD = 1.0

# --- FIN DE LA CONFIGURACIÓN ---


# --- PASO 1: OMITIDO (Tus archivos ya están organizados) ---
print("--- PASO 1: OMITIDO (archivos ya organizados) ---")


# --- PASO 2: CREAR PARCHES (TILING) ---

# --- Procesa el set de ENTRENAMIENTO ---
print("\n--- PASO 2: PROCESANDO SET DE ENTRENAMIENTO ---")
create_tiles(
    image_dir=os.path.join(INPUT_SOURCE_PATH, 'train', 'images'),
    mask_dir=os.path.join(INPUT_SOURCE_PATH, 'train', 'gt'),
    output_img_dir=os.path.join(OUTPUT_TILES_PATH, 'train', 'images'),
    output_mask_dir=os.path.join(OUTPUT_TILES_PATH, 'train', 'masks'),
    tile_size=TILE_SIZE,
    overlap=OVERLAP,
    empty_threshold=EMPTY_THRESHOLD,
    full_threshold=FULL_THRESHOLD
)

# --- Procesa el set de VALIDACIÓN ---
print("\n--- PASO 2: PROCESANDO SET DE VALIDACIÓN ---")
create_tiles(
    image_dir=os.path.join(INPUT_SOURCE_PATH, 'validation', 'images'),
    mask_dir=os.path.join(INPUT_SOURCE_PATH, 'validation', 'gt'),
    output_img_dir=os.path.join(OUTPUT_TILES_PATH, 'validation', 'images'),
    output_mask_dir=os.path.join(OUTPUT_TILES_PATH, 'validation', 'masks'),
    tile_size=TILE_SIZE,
    overlap=OVERLAP,
    empty_threshold=EMPTY_THRESHOLD,
    full_threshold=FULL_THRESHOLD
)

print("\n--- PROCESO DE TILING COMPLETADO. ---")
print(f"Tus parches están listos en: {OUTPUT_TILES_PATH}")