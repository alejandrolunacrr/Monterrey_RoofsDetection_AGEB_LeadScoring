import os
import numpy as np
import rasterio
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt  # <-- ¡NUEVO! Para graficar

# --- Bibliotecas clave ---
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler


# --- 1. CONFIGURACIÓN PRINCIPAL ---
class CONFIG:
    DATA_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\AerialImageDataset\traintiled512"
    BEST_MODEL_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_Aerial.pth"
    CHECKPOINT_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_Aerial_CHECKPOINT.pth"
    
    # --- ¡NUEVO! Ruta para guardar la gráfica ---
    PLOT_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\training_plots.png"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    EPOCHS = 145
    LEARNING_RATE = 3e-5
    IMAGE_SIZE = 512
    NUM_WORKERS = 8
    PIN_MEMORY = True
    USE_AMP = True

# --- 2. CLASE DATASET ---
# (Sin cambios)
class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations
        self.image_files = os.listdir(self.image_dir)
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) 
        with rasterio.open(img_path) as src:
            image = np.transpose(src.read(), (1, 2, 0))
        with rasterio.open(mask_path) as src:
            mask = src.read(1) 
        mask = (mask > 0).astype(np.float32)
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        mask = mask.unsqueeze(0)
        return image, mask

# --- 3. AUGMENTATIONS ---
# (Sin cambios)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = A.Compose(
    [
        A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)
val_transform = A.Compose(
    [
        A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

# --- 4. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN ---
# (Sin cambios)
scaler = GradScaler(CONFIG.DEVICE, enabled=CONFIG.USE_AMP)

def train_one_epoch(loader, model, optimizer, loss_fn_bce, loss_fn_dice):
    model.train()
    loop = tqdm(loader, desc="Entrenando", leave=True)
    epoch_loss = 0.0
    for images, masks in loop:
        images = images.to(CONFIG.DEVICE, dtype=torch.float)
        masks = masks.to(CONFIG.DEVICE, dtype=torch.float)
        optimizer.zero_grad(set_to_none=True)
        with autocast(CONFIG.DEVICE, enabled=CONFIG.USE_AMP):
            predictions = model(images)
            loss_bce = loss_fn_bce(predictions, masks)
            loss_dice = loss_fn_dice(predictions, masks)
            loss = loss_bce + loss_dice
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return epoch_loss / len(loader)

def validate_one_epoch(loader, model, loss_fn_bce, loss_fn_dice):
    model.eval()
    loop = tqdm(loader, desc="Validando", leave=True)
    val_loss = 0.0
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    with torch.no_grad():
        for images, masks in loop:
            images = images.to(CONFIG.DEVICE, dtype=torch.float)
            masks = masks.to(CONFIG.DEVICE, dtype=torch.float)
            with autocast(CONFIG.DEVICE, enabled=CONFIG.USE_AMP):
                predictions = model(images)
                loss_bce = loss_fn_bce(predictions, masks)
                loss_dice = loss_fn_dice(predictions, masks)
                loss = loss_bce + loss_dice
            val_loss += loss.item()
            tp, fp, fn, tn = smp.metrics.get_stats(
                predictions, masks.long(), mode='binary', threshold=0.5
            )
            total_tp += tp.sum(); total_fp += fp.sum(); total_fn += fn.sum(); total_tn += tn.sum()
    avg_loss = val_loss / len(loader)
    avg_iou = smp.metrics.iou_score(
        total_tp, total_fp, total_fn, total_tn, reduction='micro'
    )
    return avg_loss, avg_iou.item()

# --- ¡NUEVO! FUNCIÓN PARA GUARDAR GRÁFICAS ---
def save_plots(train_loss_hist, val_loss_hist, val_iou_hist, save_path):
    """
    Guarda las gráficas de entrenamiento (Pérdida y Métrica IoU).
    """
    epochs = range(1, len(train_loss_hist) + 1)

    plt.figure(figsize=(20, 8))

    # Gráfica 1: Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_hist, 'bo-', label='Pérdida de Entrenamiento')
    plt.plot(epochs, val_loss_hist, 'ro-', label='Pérdida de Validación')
    plt.title('Pérdida (Loss) vs. Épocas')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (BCE + Dice)')
    plt.legend()
    plt.grid(True)

    # Gráfica 2: Métrica (IoU)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_iou_hist, 'go-', label='IoU de Validación')
    plt.title('Métrica (IoU) vs. Épocas')
    plt.xlabel('Época')
    plt.ylabel('Intersection over Union (IoU)')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Resultados del Entrenamiento', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar la figura
    plt.savefig(save_path)
    print(f"Gráficas de entrenamiento guardadas en: {save_path}")
    plt.close()


# --- 5. BUCLE PRINCIPAL DE ENTRENAMIENTO ---
if __name__ == '__main__':
    print(f"--- Iniciando entrenamiento en {CONFIG.DEVICE} ---")

    # --- Setup de Datos ---
    train_dataset = BuildingDataset(
        image_dir=os.path.join(CONFIG.DATA_PATH, "train", "images"),
        mask_dir=os.path.join(CONFIG.DATA_PATH, "train", "masks"),
        augmentations=train_transform
    )
    val_dataset = BuildingDataset(
        image_dir=os.path.join(CONFIG.DATA_PATH, "validation", "images"),
        mask_dir=os.path.join(CONFIG.DATA_PATH, "validation", "masks"),
        augmentations=val_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, 
        num_workers=CONFIG.NUM_WORKERS, pin_memory=CONFIG.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, 
        num_workers=CONFIG.NUM_WORKERS, pin_memory=CONFIG.PIN_MEMORY
    )
    print(f"Datos cargados: {len(train_dataset)} train, {len(val_dataset)} val.")

    # --- Setup de Modelo y Optimizador ---
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation="sigmoid"
    ).to(CONFIG.DEVICE)
    loss_fn_dice = smp.losses.DiceLoss(mode='binary').to(CONFIG.DEVICE)
    loss_fn_bce = smp.losses.SoftBCEWithLogitsLoss().to(CONFIG.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
    
    # --- ¡NUEVO! Listas para guardar el historial ---
    train_loss_history = []
    val_loss_history = []
    val_iou_history = []
    
    # --- Lógica para Resumir ---
    start_epoch = 0
    best_iou = -1.0
    
    if os.path.exists(CONFIG.CHECKPOINT_PATH):
        print(f"¡Checkpoint encontrado! Cargando desde {CONFIG.CHECKPOINT_PATH}")
        checkpoint = torch.load(CONFIG.CHECKPOINT_PATH, map_location=torch.device('cpu')) 
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # <-- CARGA EL LR ANTIGUO
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']
        
        # --- ¡ESTAS LÍNEAS SON NUEVAS Y CRUCIALES! ---
        # Forzamos el nuevo Learning Rate en el optimizador cargado
        print(f"¡ACTUALIZANDO LEARNING RATE A: {CONFIG.LEARNING_RATE}!")
        for param_group in optimizer.param_groups:
            param_group['lr'] = CONFIG.LEARNING_RATE
        # --- FIN DE LAS LÍNEAS NUEVAS ---
            
        # ¡NUEVO! Cargar el historial guardado
        if 'history' in checkpoint:
              train_loss_history = checkpoint['history']['train_loss']
              val_loss_history = checkpoint['history']['val_loss']
              val_iou_history = checkpoint['history']['val_iou']
        
        model.to(CONFIG.DEVICE) 
        print(f"Resumiendo desde la época {start_epoch + 1}. Mejor IoU anterior: {best_iou:.4f}")
    else:
        print("No se encontró checkpoint. Empezando de cero.")
    
    # --- Bucle de Entrenamiento ---
    for epoch in range(start_epoch, CONFIG.EPOCHS):
        print(f"\n--- Época {epoch+1}/{CONFIG.EPOCHS} ---")
        
        train_loss = train_one_epoch(
            train_loader, model, optimizer, loss_fn_bce, loss_fn_dice
        )
        val_loss, val_iou = validate_one_epoch(
            val_loader, model, loss_fn_bce, loss_fn_dice
        )
        
        print(f"Pérdida (Train): {train_loss:.4f}")
        print(f"Pérdida (Val):   {val_loss:.4f} | IoU (Val): {val_iou:.4f}")
        
        # ¡NUEVO! Guardar historial de esta época
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_iou_history.append(val_iou)

        # --- Lógica de Guardado ---
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), CONFIG.BEST_MODEL_PATH)
            print(f"¡Nuevo mejor modelo guardado! (IoU: {best_iou:.4f})")
            
        checkpoint = {
            'epoch': epoch + 1, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_iou': best_iou,
            # ¡NUEVO! Guardar el historial en el checkpoint
            'history': { 
                'train_loss': train_loss_history,
                'val_loss': val_loss_history,
                'val_iou': val_iou_history
            }
        }
        torch.save(checkpoint, CONFIG.CHECKPOINT_PATH)
        print(f"Checkpoint de la época {epoch+1} guardado en {CONFIG.CHECKPOINT_PATH}")

    print("--- ENTRENAMIENTO COMPLETADO ---")
    
    # --- ¡NUEVO! Generar gráficas al finalizar ---
    if train_loss_history: # Asegurarse de que no esté vacío
        print("Generando gráficas de entrenamiento...")
        save_plots(train_loss_history, val_loss_history, val_iou_history, CONFIG.PLOT_PATH)
    else:
        print("No hay historial para graficar.")