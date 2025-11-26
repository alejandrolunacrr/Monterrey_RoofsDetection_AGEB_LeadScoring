import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# ==================== CONFIGURACIÓN ====================
CARPETA_ORIGINAL = r'C:\Users\aleja\Desktop\google_earths\CNN\predicciones_monterrey_512'
CARPETA_PROCESADA = r'C:\Users\aleja\Desktop\google_earths\CNN\predicciones_monterrey_512_NOM_AJUSTADA'
# =======================================================

def comparar_solo_con_techo():
    print(f"Buscando una imagen válida (con techo detectado)...")
    
    # 1. Listar todas las imágenes procesadas
    archivos_validos = []
    for root, dirs, files in os.walk(CARPETA_PROCESADA):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                archivos_validos.append(os.path.join(root, file))
    
    if not archivos_validos:
        print("Error: No se encontraron imágenes en la carpeta procesada.")
        return

    # 2. Revolver la lista aleatoriamente para no ver siempre las mismas
    random.shuffle(archivos_validos)
    
    imagen_encontrada = False

    # 3. Iterar hasta encontrar una que NO sea negra
    for ruta_proc in archivos_validos:
        
        # Cargar imagen procesada para verificar si tiene contenido
        img_proc = cv2.imread(ruta_proc, 0)
        
        # --- FILTRO CLAVE: ¿Tiene pixeles blancos? ---
        if cv2.countNonZero(img_proc) > 0:
            # ¡Encontramos una con techo!
            
            # Reconstruir ruta original
            ruta_relativa = os.path.relpath(ruta_proc, CARPETA_PROCESADA)
            ruta_orig = os.path.join(CARPETA_ORIGINAL, ruta_relativa)
            
            img_orig = cv2.imread(ruta_orig, 0)
            
            if img_orig is None:
                continue # Si no encuentra la original por error, busca otra

            # Asegurar binarias
            _, img_orig = cv2.threshold(img_orig, 127, 255, cv2.THRESH_BINARY)
            _, img_proc = cv2.threshold(img_proc, 127, 255, cv2.THRESH_BINARY)

            # Calcular diferencia (lo rascado)
            diff = cv2.subtract(img_orig, img_proc)

            # === VISUALIZACIÓN ===
            h, w = img_orig.shape
            vis_borde = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Rojo = Borde eliminado (NOM)
            vis_borde[diff == 255] = [255, 0, 0] 
            # Gris = Techo útil restante
            vis_borde[img_proc == 255] = [200, 200, 200] 

            print(f"MOSTRANDO: {os.path.basename(ruta_proc)}")
            print(f"Original: {cv2.countNonZero(img_orig)} px | Ajustada: {cv2.countNonZero(img_proc)} px")

            plt.figure(figsize=(15, 6))

            plt.subplot(1, 3, 1)
            plt.title("Original")
            plt.imshow(img_orig, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Ajustada NOM (-2px)")
            plt.imshow(img_proc, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Diferencia (Rojo = Eliminado)")
            plt.imshow(vis_borde)
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            
            imagen_encontrada = True
            break # Romper el ciclo, ya mostramos una.
    
    if not imagen_encontrada:
        print("Se revisaron todas las imágenes y ninguna tiene techo detectado (todas son negras).")

if __name__ == "__main__":
    comparar_solo_con_techo()