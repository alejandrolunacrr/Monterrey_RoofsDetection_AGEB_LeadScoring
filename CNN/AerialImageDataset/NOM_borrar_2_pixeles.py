import os
import cv2
import numpy as np
import rasterio

# ============================
# CONFIGURACION
# ============================
CARPETA_ENTRADA = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_predicciones_mascaras\predicciones_monterrey_512"
CARPETA_SALIDA = r"C:\Users\aleja\Desktop\google_earths\CNN\CNN_predicciones_mascaras\predicciones_monterrey_SOLIDOS"
PIXELS_A_RASCAR = 2
# ============================

def rascar_fondo_solido():
    kernel = np.ones((3, 3), np.uint8)
    contador = 0
    print("\n=== INICIANDO RASCADO (FONDO NEGRO SÓLIDO) ===")

    for root, dirs, files in os.walk(CARPETA_ENTRADA):
        for file in files:
            if not file.lower().endswith((".tif", ".tiff")):
                continue

            ruta_original = os.path.join(root, file)

            with rasterio.open(ruta_original) as src:
                img = src.read(1)
                profile = src.profile.copy()

            # 1. BINARIZAR (Seguridad)
            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # 2. RASCAR (Erosión Directa)
            # Casas Blancas (255) se hacen más chicas. Fondo Negro (0) crece.
            img_rascada = cv2.erode(img_bin, kernel, iterations=PIXELS_A_RASCAR)

            # 3. RE-BINARIZAR FINAL
            _, img_final = cv2.threshold(img_rascada, 127, 255, cv2.THRESH_BINARY)

            # 4. GUARDAR
            rel_path = os.path.relpath(root, CARPETA_ENTRADA)
            carpeta_destino = os.path.join(CARPETA_SALIDA, rel_path)
            os.makedirs(carpeta_destino, exist_ok=True)
            ruta_salida = os.path.join(carpeta_destino, file)

            # --- CAMBIO CRÍTICO AQUÍ ---
            # Ponemos nodata=None. 
            # Esto obliga a que el 0 sea un valor numérico real (Color Negro), no "vacío".
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw',
                nodata=None  # <--- ESTO ARREGLA TU PROBLEMA VISUAL
            )

            with rasterio.open(ruta_salida, 'w', **profile) as dst:
                dst.write(img_final.astype(np.uint8), 1)

            contador += 1
            if contador % 100 == 0:
                print(f"Procesadas {contador} imágenes...")

    print("=== LISTO ===")
    print(f"Las imágenes ahora tienen fondo negro SÓLIDO (Valor 0).")

if __name__ == "__main__":
    rascar_fondo_solido()