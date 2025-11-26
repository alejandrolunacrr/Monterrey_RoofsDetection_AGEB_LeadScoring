import os
import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import random

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

# Rutas exactas (basadas en tu código anterior)
VAL_IMG_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\Finetunning_Dataset\DATASET_FINAL_SPLIT\val\images"
VAL_MASK_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\Finetunning_Dataset\DATASET_FINAL_SPLIT\val\masks"
MODEL_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_Aerial_finetunned.pth"

# Dispositivo (GPU si existe, si no CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# ==========================================
# 2. CARGAR MODELO (CORREGIDO)
# ==========================================
def load_model():
    print(f"Cargando arquitectura U-Net (ResNet34)...")
    model = smp.Unet(
        encoder_name="resnet34",
        in_channels=3,
        classes=1,
        activation="sigmoid"
    )
    
    print(f"Leyendo archivo: {MODEL_PATH}")
    # Cargar el archivo completo
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # --- CORRECCIÓN AQUÍ ---
    # Verificamos si el archivo tiene la llave 'model_state_dict' (Checkpoint completo)
    # o si son los pesos directamente.
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print(" > Detectado Checkpoint completo (epoch, optimizer, etc.). Extrayendo solo los pesos...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(" > Detectado archivo de pesos directo.")
        model.load_state_dict(checkpoint)
    # -----------------------

    model.to(DEVICE)
    model.eval()
    print("Modelo cargado exitosamente.")
    return model

# ==========================================
# 3. PREPROCESAMIENTO
# ==========================================
def get_preprocessing():
    """Mismas transformaciones que en el entrenamiento"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])

# ==========================================
# 4. FUNCIÓN DE PREDICCIÓN
# ==========================================
def predict_image(model, img_path, mask_path):
    # 1. Leer Imagen Original y Máscara Real
    with rasterio.open(img_path) as src:
        # rasterio lee (Channels, Height, Width), lo pasamos a (H, W, C) para albumentations
        image_original = np.transpose(src.read((1, 2, 3)), (1, 2, 0))
    
    with rasterio.open(mask_path) as src:
        mask_real = src.read(1)

    # 2. Preprocesar para la IA
    transform = get_preprocessing()
    augmented = transform(image=image_original)
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE) # Añadir dimensión batch [1, 3, 512, 512]

    # 3. Predicción
    with torch.no_grad():
        prediction = model(img_tensor)
        prediction = prediction.squeeze().cpu().numpy() # Quitar batch y mover a CPU
    
    # 4. Binarizar (Probabilidad > 0.5 es techo)
    mask_pred = (prediction > 0.5).astype(np.uint8)

    return image_original, mask_real, mask_pred

# ==========================================
# 5. VISUALIZACIÓN
# ==========================================
def visualize_prediction(image, mask_real, mask_pred, img_name):
    # Crear figura grande
    fig, ax = plt.subplots(1, 3, figsize=(20, 7)) 
    
    # Título General
    plt.suptitle(f"AUDITORÍA DE MODELO: {img_name}", fontsize=16, fontweight='bold')
    
    # Panel 1: Original
    ax[0].imshow(image)
    ax[0].set_title("Imagen Satelital", fontsize=14)
    ax[0].axis("off")

    # Panel 2: Verdad (Ground Truth)
    ax[1].imshow(mask_real, cmap='gray')
    ax[1].set_title("Máscara Manual (Correcta)", fontsize=14)
    ax[1].axis("off")

    # Panel 3: Predicción IA
    # Mostramos la imagen de fondo y la máscara predicha en rojo/jet semitransparente
    ax[2].imshow(image)
    ax[2].imshow(mask_pred, cmap='jet', alpha=0.5) 
    ax[2].set_title("Predicción IA (Superpuesta)", fontsize=14)
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

# ==========================================
# 6. BLOQUE PRINCIPAL (MAIN)
# ==========================================
if __name__ == "__main__":
    
    # 1. Cargar modelo con la corrección
    try:
        model = load_model()
    except Exception as e:
        print(f"\n[ERROR FATAL] No se pudo cargar el modelo: {e}")
        exit()
    
    # 2. Listar imágenes de validación
    if not os.path.exists(VAL_IMG_DIR):
        print(f"No encuentro la carpeta: {VAL_IMG_DIR}")
        exit()

    images = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.tif')]
    
    if not images:
        print("¡Error! No se encontraron imágenes .tif en la carpeta de validación.")
        exit()

    print(f"\nSe encontraron {len(images)} imágenes para auditar.")
    print("--- Presiona ENTER en la consola para ver la siguiente imagen. (Ctrl+C para salir) ---")

    # 3. Bucle infinito de visualización
    while True:
        try:
            # Seleccionar al azar
            random_img_name = random.choice(images)
            
            path_img = os.path.join(VAL_IMG_DIR, random_img_name)
            path_mask = os.path.join(VAL_MASK_DIR, random_img_name)
            
            # Verificar que exista su pareja (máscara)
            if not os.path.exists(path_mask):
                print(f"Saltando {random_img_name}: No tiene máscara en {path_mask}")
                continue

            print(f" > Procesando: {random_img_name}...")
            
            # Predecir y pintar
            img, real, pred = predict_image(model, path_img, path_mask)
            visualize_prediction(img, real, pred, random_img_name)
            
            # Pausa
            input(" [ENTER] para siguiente...")
            plt.close() # Cierra la ventana actual para abrir la siguiente limpia
            
        except KeyboardInterrupt:
            print("\nSalida solicitada por usuario.")
            break
        except Exception as e:
            print(f"Error procesando imagen: {e}")
            # Intentamos con otra
            continue