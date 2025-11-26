import torch
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== CONFIGURACIÓN ====================
CHECKPOINT_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_Aerial_CHECKPOINT.pth"
VAL_IMG_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\AerialImageDataset\traintiled512\validation\images"
VAL_MASK_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\AerialImageDataset\traintiled512\validation\masks"
OUTPUT_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
IMAGE_SIZE = 512

# --- CONFIGURACIÓN DE ESTILO (LETRAS GRANDES PARA LATEX) ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 16,              # Texto general
    'axes.titlesize': 24,         # Títulos de gráficas
    'axes.labelsize': 20,         # Ejes X e Y
    'xtick.labelsize': 16,        # Números en ejes
    'ytick.labelsize': 16,
    'legend.fontsize': 18,        # Leyenda
    'lines.linewidth': 4,         # Grosor de línea
    'lines.markersize': 10,       # Tamaño de puntos
    'figure.titlesize': 26
})
# =======================================================

# --- CLASE DATASET ---
class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.png', '.jpg'))]
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        with rasterio.open(os.path.join(self.image_dir, img_name)) as src:
            image = np.transpose(src.read(), (1, 2, 0))
        with rasterio.open(os.path.join(self.mask_dir, img_name)) as src:
            mask = src.read(1)
        mask = (mask > 0).astype(np.float32)
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        mask = mask.unsqueeze(0)
        return image, mask

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def generar_reporte_entrenamiento_hd():
    print(f"--- INICIANDO GENERACIÓN DE REPORTE DE ENTRENAMIENTO (HD) ---")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print("Error: No se encuentra el archivo .pth")
        return
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    print("Checkpoint cargado correctamente.")

    # ================= PARTE A: GRÁFICAS DE HISTORIAL (SEPARADAS) =================
    if 'history' in checkpoint:
        print("\n[1/3] Generando Curvas de Aprendizaje...")
        history = checkpoint['history']
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        val_iou = history.get('val_iou', [])
        
        if train_loss:
            epochs = range(1, len(train_loss) + 1)
            
            # --- 1. GRÁFICA DE LOSS ---
            plt.figure(figsize=(10, 8))
            plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
            plt.plot(epochs, val_loss, 'r-o', label='Val Loss')
            plt.title('Pérdida (Loss) por Época')
            plt.xlabel('Épocas')
            plt.ylabel('Loss')
            plt.legend(loc='upper right', frameon=True)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            save_path_loss = os.path.join(OUTPUT_DIR, "reporte_entrenamiento_loss.png")
            plt.savefig(save_path_loss, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   -> Guardada: {save_path_loss}")

            # --- 2. GRÁFICA DE IoU ---
            plt.figure(figsize=(10, 8))
            plt.plot(epochs, val_iou, 'g-o', label='Validation IoU')
            plt.title('Métrica (IoU) por Época')
            plt.xlabel('Épocas')
            plt.ylabel('IoU Score')
            plt.legend(loc='lower right', frameon=True)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            save_path_iou = os.path.join(OUTPUT_DIR, "reporte_entrenamiento_iou.png")
            plt.savefig(save_path_iou, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   -> Guardada: {save_path_iou}")

            # Imprimir Récords
            best_iou = max(val_iou)
            print(f"\n   [Récord] Mejor IoU: {best_iou:.4f} (Época {val_iou.index(best_iou)+1})")

        else:
            print("-> El historial está vacío.")
    else:
        print("-> No se encontró historial en el checkpoint.")

    # ================= PARTE B: MATRIZ DE CONFUSIÓN =================
    print("\n[2/3] Calculando Matriz de Confusión...")
    
    # Cargar modelo
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation="sigmoid")
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(DEVICE); model.eval()

    # DataLoader
    val_ds = BuildingDataset(VAL_IMG_DIR, VAL_MASK_DIR, augmentations=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # workers=0 para evitar error en Windows

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="   Auditando píxeles"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).long()
            preds = model(images)
            tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, mode='binary', threshold=0.5)
            total_tp += tp.sum().item(); total_fp += fp.sum().item()
            total_fn += fn.sum().item(); total_tn += tn.sum().item()

    # --- 3. GRAFICAR MATRIZ ---
    print("\n[3/3] Renderizando Matriz...")
    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    # Normalizar
    sum_axis = cm.sum(axis=1)[:, np.newaxis]
    sum_axis[sum_axis == 0] = 1
    cm_percent = cm.astype('float') / sum_axis

    plt.figure(figsize=(10, 8))
    labels = ["Fondo", "Techo"]
    
    # annot_kws={"size": 28} ES LA CLAVE PARA LOS NÚMEROS GIGANTES
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 28, "weight": "bold"}, cbar=False)
    
    plt.ylabel('Realidad', fontsize=22, labelpad=15)
    plt.xlabel('Predicción', fontsize=22, labelpad=15)
    plt.title('Matriz de Confusión (Entrenamiento)', fontsize=24, pad=20)
    
    save_path_matrix = os.path.join(OUTPUT_DIR, "reporte_entrenamiento_matriz.png")
    plt.savefig(save_path_matrix, dpi=300, bbox_inches='tight')
    plt.close()

    # Métricas texto
    recall = total_tp / (total_tp + total_fn + 1e-7)
    precision = total_tp / (total_tp + total_fp + 1e-7)

    print("\n" + "="*40)
    print("      📊 RESULTADOS FINALES")
    print("="*40)
    print(f"-> Matriz guardada en: {save_path_matrix}")
    print(f"-> Recall:    {recall:.2%}")
    print(f"-> Precision: {precision:.2%}")
    print("="*40)

if __name__ == "__main__":
    generar_reporte_entrenamiento_hd()