import os
import glob
import numpy as np
import cv2
import torch
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp

"""
========================================================================
SCRIPT DE INFERENCIA MASIVA (GENERACIÓN DE MÁSCARAS PREDICHAS)
========================================================================

DESCRIPCIÓN:
Este script utiliza el modelo U-Net previamente afinado (Fine-Tuned) para 
realizar la segmentación automática de techos en un conjunto de nuevas 
imágenes satelitales (teselas) de Monterrey.

FUNCIONALIDAD:
1. Carga el modelo optimizado (.pth) en modo de evaluación.
2. Recorre el directorio de entrada (INPUT_DIR) buscando archivos .tif.
3. Aplica una técnica de "Ventana Deslizante" (Sliding Window) para procesar
   cada imagen por parches, asegurando que se respete la resolución de 
   entrada del modelo (512x512) y gestionando el padding automáticamente.
4. Genera una máscara binaria (0 = Fondo, 255 = Techo).
5. Guarda el resultado como un archivo GeoTIFF, conservando los metadatos
   de georreferenciación (coordenadas y proyección) originales.

OBJETIVO:
Transformar las imágenes satelitales crudas en mapas binarios de techos
listos para ser vectorizados o analizados en sistemas GIS (QGIS).
========================================================================
"""

# --- 1. CONFIGURACIÓN ---

class CONFIG:
    # 1. El modelo "MEJOR" que acabas de entrenar (Esta ya está bien)
    MODEL_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\modelo_finetuned_BEST.pth"
    
    # 2. ¡CORRECCIÓN! Apunta a la carpeta con los tiles de 512x512
    INPUT_DIR = r"C:\Users\aleja\Desktop\google_earths\Monterrey_Google_Tiles_512\teselas_geotiff4_512" 
    
    # 3. (Recomendado) Cambia el nombre de la salida para no confundirte
    OUTPUT_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_predicciones_mascaras\predicciones_monterrey_512\cuadrante4"

    # 4. Parámetros (Esta ya está bien)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_SIZE = 512 # El tamaño de parche con el que entrenaste
    BATCH_SIZE = 16  # Puedes ajustar esto según tu VRAM
    
# ---------------------

# --- 2. DEFINIR MODELO Y TRANSFORMACIONES ---
# (Esta sección estaba correcta)

print(f"Usando dispositivo: {CONFIG.DEVICE}")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation="sigmoid"
).to(CONFIG.DEVICE)

print(f"Cargando modelo desde {CONFIG.MODEL_PATH}...")
model.load_state_dict(torch.load(CONFIG.MODEL_PATH, map_location=CONFIG.DEVICE))
model.eval() 

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transform = A.Compose(
    [
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

# --- 3. LÓGICA DE PREDICCIÓN (SLIDING WINDOW) ---

def predict_large_tile(model, tile_path, output_path, transform, patch_size, batch_size, device):
    """
    Predice en un GeoTIFF grande usando un método de "ventana deslizante" (patching).
    *** VERSIÓN CORREGIDA ***
    """
    try:
        # --- 1. Abrir la imagen grande y leer metadatos ---
        with rasterio.open(tile_path) as src:
            meta = src.meta.copy()
            
            try:
                image = np.transpose(src.read((1, 2, 3)), (1, 2, 0))
            except rasterio.errors.RasterioIOError:
                print(f"  Advertencia: No se pudieron leer 3 bandas de {tile_path}. Omitiendo.")
                return

            orig_h, orig_w = image.shape[:2]

            # --- 2. Preparar padding ---
            pad_h = (patch_size - orig_h % patch_size) % patch_size
            pad_w = (patch_size - orig_w % patch_size) % patch_size
            
            padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            new_h, new_w = padded_image.shape[:2]

            # --- 3. Crear canvas para la máscara de salida ---
            padded_mask = np.zeros((new_h, new_w), dtype=np.float32)
            
            patches = []
            coords = []

            # --- 4. (CORRECCIÓN) Crear todos los parches y coordenadas ---
            # ¡Este bucle faltaba en tu script!
            for r in range(0, new_h, patch_size):
                for c in range(0, new_w, patch_size):
                    # Cortar el parche
                    patch = padded_image[r:r+patch_size, c:c+patch_size]
                    # Aplicar transformaciones (Normalize + ToTensorV2)
                    patches.append(transform(image=patch)['image'])
                    # Guardar las coordenadas para pegar el resultado
                    coords.append((r, c))

            # --- 5. Predecir en lotes (batches) para máxima velocidad ---
            with torch.no_grad():
                # (Esta parte estaba bien, pero no se ejecutaba porque 'patches' estaba vacío)
                # He añadido tqdm aquí para que veas el progreso por parche
                for i in tqdm(range(0, len(patches), batch_size), desc="  Prediciendo parches", leave=False):
                    batch_patches = patches[i:i+batch_size]
                    batch_coords = coords[i:i+batch_size]
                    
                    batch_tensor = torch.stack(batch_patches).to(device)
                    
                    pred_batch = model(batch_tensor)
                    
                    pred_batch_np = pred_batch.squeeze(1).cpu().numpy()
                    
                    for j, (r, c) in enumerate(batch_coords):
                        padded_mask[r:r+patch_size, c:c+patch_size] = pred_batch_np[j]

            # --- 6. Quitar el padding y binarizar ---
            final_mask_prob = padded_mask[0:orig_h, 0:orig_w]
            final_mask_uint8 = (final_mask_prob > 0.5).astype(np.uint8) * 255
            
            # --- 7. Guardar la máscara georeferenciada ---
            # (El bloque erróneo se ha eliminado. Esta es la forma correcta)
            meta.update(count=1, dtype='uint8', compress='lzw', nodata=None)
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(final_mask_uint8, 1)

    except Exception as e:
        print(f"Error procesando {os.path.basename(tile_path)}: {e}")

# --- 4. BUCLE PRINCIPAL DE EJECUCIÓN ---

if __name__ == "__main__":
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(CONFIG.INPUT_DIR, "*.tif"))
    
    if not image_paths:
        print(f"¡ERROR! No se encontraron archivos .tif en {CONFIG.INPUT_DIR}")
        print("Asegúrate de que la ruta sea correcta y tus tiles de Monterrey estén ahí.")
    else:
        print(f"Iniciando predicción en {len(image_paths)} tiles...")
        
        for tile_path in tqdm(image_paths, desc="Procesando Tiles de Monterrey"):
            file_name = os.path.basename(tile_path)
            output_path = os.path.join(CONFIG.OUTPUT_DIR, file_name.replace(".tif", "_MASCARA.tif"))
            
            predict_large_tile(
                model, 
                tile_path, 
                output_path, 
                val_transform, 
                CONFIG.IMAGE_SIZE, 
                CONFIG.BATCH_SIZE, 
                CONFIG.DEVICE
            )
            
        print(f"\n¡Predicción completada! Máscaras guardadas en: {CONFIG.OUTPUT_DIR}")