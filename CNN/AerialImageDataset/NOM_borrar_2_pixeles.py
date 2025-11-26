import cv2
import numpy as np
import os

# ==================== CONFIGURACIÓN ====================
# 1. Dónde están tus máscaras originales
CARPETA_ENTRADA = r'C:\Users\aleja\Desktop\google_earths\CNN\predicciones_monterrey_512'

# 2. Dónde quieres guardar las nuevas máscaras "rascadas"
# Se creará automáticamente esta carpeta
CARPETA_SALIDA = r'C:\Users\aleja\Desktop\google_earths\CNN\predicciones_monterrey_512_NOM_AJUSTADA'

# 3. Cuántos pixeles quieres quitar
PIXELES_A_RASCAR = 2
# =======================================================

def rascar_bordes():
    # Creamos el elemento estructurante básico de 3x3
    # Al aplicarlo 1 vez, quita 1 pixel. Al aplicarlo N veces, quita N pixeles.
    kernel = np.ones((3,3), np.uint8)
    
    contador = 0
    
    print(f"--- INICIANDO PROCESO DE AJUSTE NOM ---")
    print(f"Origen: {CARPETA_ENTRADA}")
    print(f"Destino: {CARPETA_SALIDA}")
    print(f"Se eliminarán {PIXELES_A_RASCAR} pixeles del borde de cada techo.")

    # Recorremos todas las carpetas (cuadrante2, 3, 4)
    for root, dirs, files in os.walk(CARPETA_ENTRADA):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                
                # Ruta completa del archivo original
                ruta_original = os.path.join(root, file)
                
                # Leemos la imagen en Blanco y Negro (0)
                img = cv2.imread(ruta_original, 0)
                
                if img is None:
                    continue
                
                # Aseguramos que sea binaria pura (0 y 255) para evitar bordes grises
                _, img_binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                
                # === LA MAGIA: EROSIÓN ===
                # iterations=PIXELES_A_RASCAR hace que el kernel de 3x3 pase 2 veces,
                # pelando exactamente 1 pixel en cada pasada = 2 pixeles total.
                img_ajustada = cv2.erode(img_binaria, kernel, iterations=PIXELES_A_RASCAR)
                
                # === GUARDADO ===
                # Replicamos la estructura de carpetas (ej. crear carpeta "cuadrante2" en el destino)
                rel_path = os.path.relpath(root, CARPETA_ENTRADA)
                carpeta_destino_final = os.path.join(CARPETA_SALIDA, rel_path)
                
                if not os.path.exists(carpeta_destino_final):
                    os.makedirs(carpeta_destino_final)
                
                ruta_guardado = os.path.join(carpeta_destino_final, file)
                
                # Guardamos la imagen procesada
                cv2.imwrite(ruta_guardado, img_ajustada)
                
                contador += 1
                if contador % 100 == 0:
                    print(f"Procesadas {contador} imágenes...")

    print(f"--- LISTO ---")
    print(f"Total de imágenes ajustadas a NOM: {contador}")
    print(f"Guardadas en: {CARPETA_SALIDA}")

if __name__ == "__main__":
    rascar_bordes()