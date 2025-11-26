import os
import shutil
import random
from pathlib import Path

"""
========================================================================
SCRIPT DE PREPARACIÓN DE DATASET PARA FINE-TUNING (TRAIN/VAL SPLIT)
========================================================================

DESCRIPCIÓN:
Este script se encarga de consolidar y organizar el dataset manual creado
a partir de las teselas de Google Earth para el proceso de Fine-Tuning.

FUNCIONALIDAD:
1. Recorre las carpetas filtradas por densidad ("top_20_techos_...").
2. Identifica los pares de imágenes y sus correspondientes máscaras manuales
   ubicadas en 'masks_MANUALES'.
3. Realiza una división aleatoria (shuffle) de los datos:
   - 80% para Entrenamiento (Train)
   - 20% para Validación (Val)
4. Copia y renombra los archivos a una estructura de carpetas estándar
   compatible con PyTorch/U-Net:
      DATASET_FINAL_SPLIT/
      ├── train/
      │   ├── images/
      │   └── masks/
      └── val/
          ├── images/
          └── masks/

OBJETIVO:
Generar un conjunto de datos limpio, balanceado y estructurado, listo para
alimentar la etapa de ajuste fino (fine-tuning) del modelo de segmentación.
========================================================================
"""

# --- CONFIGURACIÓN ---
# Carpeta raíz donde están las carpetas "top_20_techos_..."
INPUT_ROOT = r"C:\Users\aleja\Desktop\google_earths\CNN\Filtros"

# Donde se guardará el dataset final listo para entrenar
OUTPUT_ROOT = r"C:\Users\aleja\Desktop\google_earths\CNN\DATASET_FINAL_SPLIT"

# Porcentaje de validación (0.2 = 20%)
VAL_SPLIT = 0.2

# Lista de tus carpetas (para asegurar el orden y que existan)
CARPETAS_ORIGEN = [
    "top_20_techos_10-20_porciento",
    "top_20_techos_20-30_porciento",
    "top_20_techos_30-40_porciento",
    "top_20_techos_40-50_porciento",
    "top_20_techos_50-60_porciento",
    "top_20_techos_60-70_porciento",
    "top_20_techos_70-80_porciento",
    "top_20_techos_80-90_porciento"
]
# ---------------------

def create_structure():
    """Crea las carpetas train/images, train/masks, val/images, val/masks"""
    for subset in ['train', 'val']:
        for tipo in ['images', 'masks']:
            path = os.path.join(OUTPUT_ROOT, subset, tipo)
            os.makedirs(path, exist_ok=True)
    print(f"Estructura de carpetas creada en: {OUTPUT_ROOT}")

def get_all_pairs():
    """Recorre las carpetas y busca pares imagen-máscara válidos."""
    dataset = [] # Lista de tuplas: (ruta_imagen, ruta_mascara, nombre_archivo, prefijo_carpeta)
    
    print("\n--- Buscando archivos ---")
    
    for carpeta in CARPETAS_ORIGEN:
        path_images = os.path.join(INPUT_ROOT, carpeta, "images")
        path_masks = os.path.join(INPUT_ROOT, carpeta, "masks_MANUALES")
        
        if not os.path.exists(path_masks):
            print(f"[!] Saltando {carpeta}: No existe carpeta 'masks_MANUALES'")
            continue
            
        # Buscar máscaras (que son lo que acabas de crear)
        masks_files = [f for f in os.listdir(path_masks) if f.endswith('.tif')]
        
        for m_file in masks_files:
            m_path = os.path.join(path_masks, m_file)
            i_path = os.path.join(path_images, m_file) # Asumimos mismo nombre
            
            if os.path.exists(i_path):
                # Guardamos datos necesarios para copiar después
                # Usamos el nombre de la carpeta como prefijo para evitar archivos duplicados
                dataset.append({
                    'src_img': i_path,
                    'src_mask': m_path,
                    'filename': m_file,
                    'prefix': carpeta.replace("top_20_techos_", "") # Ej: "10-20_porciento"
                })
            else:
                print(f"[AVISO] Máscara sin imagen original encontrada: {m_file} en {carpeta}")
    
    return dataset

def copy_files(data_list, subset_name):
    """Copia la lista de archivos a la carpeta train o val."""
    print(f"\nCopiando {len(data_list)} archivos a '{subset_name}'...")
    
    dest_img_dir = os.path.join(OUTPUT_ROOT, subset_name, "images")
    dest_mask_dir = os.path.join(OUTPUT_ROOT, subset_name, "masks")
    
    for item in data_list:
        # Crear nuevo nombre único: "10-20_porciento_nombreoriginal.tif"
        new_name = f"{item['prefix']}_{item['filename']}"
        
        # Rutas finales
        dst_img = os.path.join(dest_img_dir, new_name)
        dst_mask = os.path.join(dest_mask_dir, new_name)
        
        # Copiar
        shutil.copy2(item['src_img'], dst_img)
        shutil.copy2(item['src_mask'], dst_mask)

def main():
    # 1. Preparar
    create_structure()
    
    # 2. Recopilar todo
    full_dataset = get_all_pairs()
    total = len(full_dataset)
    
    if total == 0:
        print("¡Error! No se encontraron pares de imágenes y máscaras.")
        return

    print(f"\nTotal de pares encontrados: {total}")
    
    # 3. Barajar (Aleatoriedad importante)
    random.shuffle(full_dataset)
    
    # 4. Calcular división
    val_count = int(total * VAL_SPLIT)
    train_count = total - val_count
    
    train_set = full_dataset[:train_count]
    val_set = full_dataset[train_count:]
    
    print(f"Set de Entrenamiento (Train): {len(train_set)} imágenes (80%)")
    print(f"Set de Validación (Val):      {len(val_set)} imágenes (20%)")
    
    # 5. Copiar archivos
    copy_files(train_set, "train")
    copy_files(val_set, "val")
    
    print("\n" + "="*50)
    print("¡PROCESO COMPLETADO EXITOSAMENTE!")
    print(f"Dataset listo en: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()