import os
import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import random
import math

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
VAL_IMG_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\Finetunning_Dataset\DATASET_FINAL_SPLIT\val\images"
VAL_MASK_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\Finetunning_Dataset\DATASET_FINAL_SPLIT\val\masks"
MODEL_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_Aerial_finetunned.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# ==========================================
# 2. CARGAR MODELO
# ==========================================
def load_model():
    print(f"Cargando arquitectura U-Net (ResNet34)...")
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation="sigmoid")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

# ==========================================
# 3. HELPER: NORMALIZAR DISPLAY
# ==========================================
def normalize_for_display(img):
    img = img.astype(float)
    if img.max() == img.min():
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)

# ==========================================
# 4. PREPROCESAMIENTO
# ==========================================
def get_preprocessing():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])

# ==========================================
# 5. FUNCIÓN DE PREDICCIÓN
# ==========================================
def predict_image(model, img_path, mask_path):
    with rasterio.open(img_path) as src:
        raw_img = np.transpose(src.read((1, 2, 3)), (1, 2, 0))
        bounds = src.bounds
        res_x = src.res[0]

        if res_x < 0.1:
            factor_conversion = 111320 * math.cos(math.radians(25.6)) 
            res_metros = res_x * factor_conversion
        else:
            res_metros = res_x

    with rasterio.open(mask_path) as src:
        mask_real = src.read(1)

    transform = get_preprocessing()
    augmented = transform(image=raw_img)
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(img_tensor)
        prediction = prediction.squeeze().cpu().numpy()
    
    mask_pred = (prediction > 0.5).astype(np.uint8)
    image_vis = normalize_for_display(raw_img)
    
    return image_vis, mask_real, mask_pred, bounds, res_metros

# ==========================================
# 6. BARRA DE ESCALA (0, 20, 40m)
# ==========================================
def draw_aligned_scale_bar(ax, image_shape, pixel_size_m, max_meters=40, step_meters=20):
    height, width = image_shape[:2]
    
    pixels_per_step = step_meters / pixel_size_m
    total_width_px = max_meters / pixel_size_m
    
    # Alineación derecha con margen
    safe_right_limit = width - 15
    grid_index = int(safe_right_limit // pixels_per_step)
    
    end_x = grid_index * pixels_per_step
    start_x = end_x - total_width_px
    
    bar_height = 15 # Un poco más gruesa
    margin_y = 30
    start_y = height - margin_y - bar_height
    
    # Borde negro para el texto
    text_effects = [pe.withStroke(linewidth=3, foreground="black")]
    
    current_x = start_x
    colors = ['black', 'white', 'black', 'white'] 
    
    # Dibujar segmentos
    for i in range(0, max_meters, step_meters):
        idx = int(i / step_meters)
        color = colors[idx % len(colors)]
        
        rect = patches.Rectangle(
            (current_x, start_y), pixels_per_step, bar_height,
            linewidth=1.5, edgecolor='black', facecolor=color
        )
        ax.add_patch(rect)
        
        # TEXTO DEL NÚMERO (Tamaño aumentado)
        label = str(i)
        ax.text(current_x, start_y - 6, label, 
                color='white', ha='center', va='bottom', 
                fontsize=14, fontweight='bold', path_effects=text_effects) # fontsize 14
        
        current_x += pixels_per_step

    # TEXTO FINAL (Tamaño aumentado)
    ax.text(current_x, start_y - 6, f"{max_meters}m", 
            color='white', ha='center', va='bottom', 
            fontsize=14, fontweight='bold', path_effects=text_effects) # fontsize 14

# ==========================================
# 7. VISUALIZACIÓN FINAL (Compacta y Grande)
# ==========================================
def visualize_for_paper(image, mask_real, mask_pred, bounds, res_m, img_name, save=False):
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
    plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99)

    FONT_TITLE = 16
    
    # --- CALCULO DE GRID (20m para coincidir con escala) ---
    pixels_step = 20.0 / res_m # Grid de 20 en 20
    h, w = image.shape[:2]
    xticks = np.arange(0, w, pixels_step)
    yticks = np.arange(0, h, pixels_step)
    # ----------------------------
    
    # --- Panel A ---
    axs[0].imshow(image) 
    axs[0].set_title("a) Imagen Satelital", fontsize=FONT_TITLE, fontweight='bold', pad=8)
    
    # Escala de 0 a 40m, pasos de 20m
    draw_aligned_scale_bar(axs[0], image.shape, res_m, max_meters=40, step_meters=20)
    
    # --- Panel B ---
    axs[1].imshow(mask_real, cmap='gray')
    axs[1].set_title("b) Máscaras Manuales", fontsize=FONT_TITLE, fontweight='bold', pad=8)

    # --- Panel C ---
    axs[2].imshow(image)
    axs[2].imshow(mask_pred, cmap='jet', alpha=0.5) 
    axs[2].set_title("c) Predicción CNN", fontsize=FONT_TITLE, fontweight='bold', pad=8)

    # --- Configuración Común ---
    for ax in axs:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        
        # Grid (más sutil)
        ax.grid(color='white', linestyle='--', linewidth=1.5, alpha=0.6)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

    # --- TEXTO DE COORDENADAS (Más grande) ---
    center_lon = (bounds.left + bounds.right) / 2
    center_lat = (bounds.bottom + bounds.top) / 2
    
    texto_coords = f"Coordenadas: {center_lat:.6f}° N, {center_lon:.6f}° W"
    
    axs[1].text(0.5, -0.02, texto_coords, 
                transform=axs[1].transAxes, 
                ha='center', va='top', 
                fontsize=16, color='black', fontweight='bold') # fontsize 16 y negrita

    plt.show()
    
    if save:
        filename = f"Fig_Final_BigText_{img_name.replace('.tif', '')}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" [Guardado] {filename}")
        plt.close(fig)

# ==========================================
# 8. MAIN
# ==========================================
if __name__ == "__main__":
    try:
        model = load_model()
    except Exception as e:
        print(f"[ERROR] Modelo: {e}")
        exit()
    
    images = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.tif')]
    if not images: exit()

    print("--- LISTO ---")
    while True:
        try:
            random_img_name = random.choice(images)
            path_img = os.path.join(VAL_IMG_DIR, random_img_name)
            path_mask = os.path.join(VAL_MASK_DIR, random_img_name)
            
            if not os.path.exists(path_mask): continue

            print(f" > Procesando: {random_img_name}")
            
            img, real, pred, bounds, res_m = predict_image(model, path_img, path_mask)
            
            visualize_for_paper(img, real, pred, bounds, res_m, random_img_name, save=False)
            
            accion = input("¿Guardar? (s/n/salir): ").lower()
            if accion == 's':
                visualize_for_paper(img, real, pred, bounds, res_m, random_img_name, save=True)
            elif accion == 'salir':
                break
        except Exception as e:
            print(f"Error: {e}")
            continue