import os
import glob
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# --- 1. CONFIGURACIÓN ---

class CONFIG:
    # Carpetas de Origen
    IMAGE_SOURCE_DIR = r"C:\Users\aleja\Desktop\google_earths\teselas_geotiff2_512"
    MASK_SOURCE_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\predicciones_monterrey_512"
    
    # Carpeta base para las salidas
    OUTPUT_BASE_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\Filtros"
    
    # --- ¡Parámetros del filtro! ---
    # Cambia estos valores para cada ejecución
    MIN_THRESHOLD_PERCENT = 0.10
    MAX_THRESHOLD_PERCENT = 0.20  
    MAX_FILES_TO_COPY = 20
    
    # Sufijo de tus máscaras
    MASK_SUFFIX = "_MASCARA.tif"
    IMAGE_EXT = ".tif" # Extensión de la imagen original

# --------------------------

def filter_and_copy_tiles():
    
    # --- ¡NUEVO! Generar nombre de salida dinámico ---
    min_str = int(CONFIG.MIN_THRESHOLD_PERCENT * 100)
    max_str = int(CONFIG.MAX_THRESHOLD_PERCENT * 100)
    folder_name = f"top_{CONFIG.MAX_FILES_TO_COPY}_techos_{min_str}-{max_str}_porciento"
    OUTPUT_DIR = os.path.join(CONFIG.OUTPUT_BASE_DIR, folder_name)
    
    print(f"--- Iniciando filtro de tiles (rango {min_str}%-{max_str}%) ---")
    print(f"--- Guardando en: {OUTPUT_DIR} ---")
    
    # --- 1. Crear carpetas de destino ---
    output_img_dir = os.path.join(OUTPUT_DIR, 'images')
    output_mask_dir = os.path.join(OUTPUT_DIR, 'masks')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # --- 2. Encontrar todas las máscaras ---
    mask_files = glob.glob(os.path.join(CONFIG.MASK_SOURCE_DIR, f"*{CONFIG.MASK_SUFFIX}"))
    
    if not mask_files:
        print(f"¡Error! No se encontraron archivos de máscara en {CONFIG.MASK_SOURCE_DIR}")
        return

    print(f"Encontradas {len(mask_files)} máscaras. Analizando...")
    
    files_copied = 0
    pbar = tqdm(total=CONFIG.MAX_FILES_TO_COPY, desc="Buscando tiles en rango")
    np.random.shuffle(mask_files)
    
    # --- 3. Iterar y filtrar ---
    for mask_path in mask_files:
        if files_copied >= CONFIG.MAX_FILES_TO_COPY:
            print(f"\nSe alcanzó el límite de {CONFIG.MAX_FILES_TO_COPY} archivos.")
            break
            
        try:
            # --- 4. Calcular porcentaje (rápido) ---
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            roof_percentage = np.mean(mask == 255)
            
            # --- 5. Comprobar la condición de RANGO ---
            if (roof_percentage >= CONFIG.MIN_THRESHOLD_PERCENT) and (roof_percentage <= CONFIG.MAX_THRESHOLD_PERCENT):
                
                # --- 6. ¡LÓGICA CORREGIDA! Obtener el nombre de la imagen ---
                mask_name = os.path.basename(mask_path)
                # Simplemente quita el sufijo para obtener el nombre de la imagen
                image_name = mask_name.replace(CONFIG.MASK_SUFFIX, CONFIG.IMAGE_EXT)
                
                image_path = os.path.join(CONFIG.IMAGE_SOURCE_DIR, image_name)
                
                # --- 7. Verificar y Copiar ---
                if os.path.exists(image_path):
                    dest_mask_path = os.path.join(output_mask_dir, mask_name)
                    dest_image_path = os.path.join(output_img_dir, image_name)
                    
                    shutil.copyfile(mask_path, dest_mask_path)
                    shutil.copyfile(image_path, dest_image_path)
                    
                    files_copied += 1
                    pbar.update(1)
                else:
                    # Si ves esto, significa que el nombre de la imagen y la máscara no coinciden
                    print(f"\nAdvertencia: Máscara {mask_name} encontrada, pero falta la imagen {image_name}")
                    pass
                    
        except Exception as e:
            print(f"\nError procesando {mask_path}: {e}")
            
    pbar.close()
    print(f"\n--- Proceso completado ---")
    print(f"Se copiaron {files_copied} pares de imágenes/máscaras en: {OUTPUT_DIR}")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    filter_and_copy_tiles()