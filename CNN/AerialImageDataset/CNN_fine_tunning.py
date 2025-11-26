import os
import numpy as np
import rasterio
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler

# --- 1. CONFIGURACIÓN PARA FINE-TUNING ---
class CONFIG:
    # Rutas Nuevas (Dataset Manual y Modelos Nuevos)
    DATA_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\DATASET_FINAL_SPLIT"
    
    # ¡IMPORTANTE! Cargamos el MEJOR modelo del entrenamiento anterior
    PRETRAINED_MODEL_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_Aerial.pth"
    
    # Guardaremos con nombres nuevos para no sobrescribir el original
    BEST_MODEL_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_Aerial_finetunned.pth"
    CHECKPOINT_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_Aerial_finetunned_CHECKPOINT.pth"
    PLOT_PATH = r"C:\Users\aleja\Desktop\google_earths\CNN\finetuning_plots.png"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4         # Puedes probar bajar a 2 si te falta VRAM
    EPOCHS = 10            # Menos épocas, ya que es ajuste fino
    LEARNING_RATE = 1e-5   # ¡MUY BAJO! (Antes era 3e-5, ahora mas lento y preciso)
    IMAGE_SIZE = 512
    NUM_WORKERS = 4        # Ajustado para Windows (a veces 8 da problemas, probar 4)
    PIN_MEMORY = True
    USE_AMP = True

# --- 2. CLASE DATASET (Igual) ---
class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.tif')] # Filtro extra seguridad

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        with rasterio.open(img_path) as src:
            image = np.transpose(src.read((1,2,3)), (1, 2, 0)) # Leer solo bandas RGB por seguridad
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            
        mask = (mask > 0).astype(np.float32)
        
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        mask = mask.unsqueeze(0)
        return image, mask

# --- 3. AUGMENTATIONS (Igual) ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = A.Compose([
    A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
    A.HorizontalFlip(p=0.5), 
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5), # Agregué rotación para dar más variedad a los pocos datos
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2(),
])

# --- 4. FUNCIONES (Igual) ---
scaler = GradScaler(CONFIG.DEVICE, enabled=CONFIG.USE_AMP)

def train_one_epoch(loader, model, optimizer, loss_fn_bce, loss_fn_dice):
    model.train()
    loop = tqdm(loader, desc="Fine-Tuning", leave=True)
    epoch_loss = 0.0
    for images, masks in loop:
        images = images.to(CONFIG.DEVICE, dtype=torch.float)
        masks = masks.to(CONFIG.DEVICE, dtype=torch.float)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(CONFIG.DEVICE, enabled=CONFIG.USE_AMP):
            predictions = model(images)
            loss_bce = loss_fn_bce(predictions, masks)
            loss_dice = loss_fn_dice(predictions, masks)
            loss = loss_bce + loss_dice # Peso igual
            
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
            
            # Métricas
            tp, fp, fn, tn = smp.metrics.get_stats(
                predictions, masks.long(), mode='binary', threshold=0.5
            )
            total_tp += tp.sum(); total_fp += fp.sum(); total_fn += fn.sum(); total_tn += tn.sum()
            
    avg_loss = val_loss / len(loader)
    avg_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
    return avg_loss, avg_iou.item()

def save_plots(train_hist, val_hist, iou_hist, path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Val Loss')
    plt.title('Loss during Fine-Tuning')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(iou_hist, label='Val IoU', color='green')
    plt.title('IoU Improvement')
    plt.legend()
    plt.savefig(path)
    plt.close()

# --- 5. PRINCIPAL (LÓGICA DE REANUDACIÓN INTELIGENTE) ---
if __name__ == '__main__':
    print(f"--- INICIANDO FINE-TUNING (LR={CONFIG.LEARNING_RATE}) ---")
    
    # 1. Cargar Datasets
    train_ds = BuildingDataset(
        os.path.join(CONFIG.DATA_PATH, "train", "images"),
        os.path.join(CONFIG.DATA_PATH, "train", "masks"),
        augmentations=train_transform
    )
    val_ds = BuildingDataset(
        os.path.join(CONFIG.DATA_PATH, "val", "images"),
        os.path.join(CONFIG.DATA_PATH, "val", "masks"),
        augmentations=val_transform
    )
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)
    
    print(f"Datos Manuales: {len(train_ds)} Train | {len(val_ds)} Val")

    # 2. Instanciar Modelo, Loss y Optimizador
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid").to(CONFIG.DEVICE)
    loss_fn_bce = smp.losses.SoftBCEWithLogitsLoss().to(CONFIG.DEVICE)
    loss_fn_dice = smp.losses.DiceLoss(mode='binary').to(CONFIG.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)

    # 3. VARIABLES DE ESTADO INICIALES
    start_epoch = 0
    best_iou = 0.0
    train_loss_hist = []
    val_loss_hist = []
    val_iou_hist = []

    # ==============================================================================
    # 4. LÓGICA DE CARGA DE PESOS (PRIORIDAD: CHECKPOINT > PRETRAINED)
    # ==============================================================================
    
    # CASO A: ¿Existe un checkpoint de Fine-Tuning previo? (REANUDAR)
    if os.path.exists(CONFIG.CHECKPOINT_PATH):
        print(f"✅ ¡ENCONTRADO CHECKPOINT DE FINE-TUNING!: {CONFIG.CHECKPOINT_PATH}")
        print("-> Cargando estado completo para reanudar...")
        
        checkpoint = torch.load(CONFIG.CHECKPOINT_PATH, map_location=CONFIG.DEVICE)
        
        # Cargar todo el estado
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint.get('scaler_state_dict', scaler.state_dict()))
        
        start_epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']
        
        # Recuperar historial si existe
        if 'history' in checkpoint:
            hist = checkpoint['history']
            train_loss_hist = hist.get('train_loss', [])
            val_loss_hist = hist.get('val_loss', [])
            val_iou_hist = hist.get('val_iou', [])
            
        print(f"-> Sistema listo. Reanudando desde la ÉPOCA {start_epoch + 1}")

    # CASO B: No hay checkpoint, ¿Existe el modelo base? (EMPEZAR DE CERO)
    elif os.path.exists(CONFIG.PRETRAINED_MODEL_PATH):
        print(f"⚠️ No hay checkpoint de fine-tuning. Buscando modelo base...")
        print(f"-> Cargando modelo base: {CONFIG.PRETRAINED_MODEL_PATH}")
        
        base_checkpoint = torch.load(CONFIG.PRETRAINED_MODEL_PATH, map_location=CONFIG.DEVICE)
        
        # Manejo de diccionarios vs pesos crudos
        if 'model_state_dict' in base_checkpoint:
            model.load_state_dict(base_checkpoint['model_state_dict'])
            # OJO: Si el modelo base tiene 'epoch', PODRÍAMOS usarla, pero
            # para Fine-Tuning suele ser mejor reiniciar el contador visual a 1
            # o continuarlo. Aquí lo REINICIAMOS a 0 para el nuevo dataset.
            print("-> Pesos del modelo base cargados correctamente.")
        else:
            model.load_state_dict(base_checkpoint)
            print("-> Pesos crudos cargados correctamente.")
            
        print("-> Iniciando Fine-Tuning desde la ÉPOCA 1")
        
    else:
        raise FileNotFoundError("❌ ERROR CRÍTICO: No se encontró ni el Checkpoint ni el Modelo Base.")

    # ==============================================================================
    # 5. BUCLE DE ENTRENAMIENTO
    # ==============================================================================
    
    # Definimos hasta qué época llegar. 
    # Si reanuda en la 10 y CONFIG.EPOCHS es 50, llegará a la 60.
    total_target_epochs = start_epoch + CONFIG.EPOCHS
    
    for epoch in range(start_epoch, total_target_epochs):
        print(f"\nEpoch {epoch+1}/{total_target_epochs}")
        
        t_loss = train_one_epoch(train_loader, model, optimizer, loss_fn_bce, loss_fn_dice)
        v_loss, v_iou = validate_one_epoch(val_loader, model, loss_fn_bce, loss_fn_dice)
        
        print(f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val IoU: {v_iou:.4f}")
        
        train_loss_hist.append(t_loss)
        val_loss_hist.append(v_loss)
        val_iou_hist.append(v_iou)

        # Guardar MEJOR modelo
        if v_iou > best_iou:
            best_iou = v_iou
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'history': {
                    'train_loss': train_loss_hist,
                    'val_loss': val_loss_hist,
                    'val_iou': val_iou_hist
                }
            }
            torch.save(checkpoint_data, CONFIG.BEST_MODEL_PATH)
            print(f"★ ¡Nuevo Récord! Modelo guardado (IoU: {best_iou:.4f})")

        # Guardar CHECKPOINT (Siempre, para poder reanudar si se va la luz)
        checkpoint_current = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), # ¡Guarda el estado del AdamW!
            'scaler_state_dict': scaler.state_dict(),
            'best_iou': best_iou,
            'history': {
                'train_loss': train_loss_hist,
                'val_loss': val_loss_hist,
                'val_iou': val_iou_hist
            }
        }
        torch.save(checkpoint_current, CONFIG.CHECKPOINT_PATH)
        
        # Guardar gráficas
        save_plots(train_loss_hist, val_loss_hist, val_iou_hist, CONFIG.PLOT_PATH)

    print(f"--- FINE TUNING TERMINADO (Época {total_target_epochs}) ---")