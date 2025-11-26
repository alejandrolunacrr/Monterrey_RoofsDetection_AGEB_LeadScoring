import os
import glob
import rasterio
from rasterio.plot import plotting_extent
import numpy as np
import matplotlib.pyplot as plt
# ¡NUEVA IMPORTACIÓN para dibujar la cuadrícula!
from matplotlib.patches import Rectangle 

# 1. Definir la ruta a la carpeta que contiene las teselas
folder_path = r'C:\Users\aleja\Desktop\google_earths\manejo_imagenes_geotiff\seleccion\seleccion4'

# Verificar si la carpeta existe
if not os.path.isdir(folder_path):
    print(f"Error: La carpeta '{folder_path}' no fue encontrada.")
    exit()

# 2. Encontrar todos los archivos GeoTIFF (.tif)
search_criteria = "*.tif"
tiff_files = glob.glob(os.path.join(folder_path, search_criteria))
tiff_files.sort() # Opcional: ordenar los archivos alfabéticamente

if not tiff_files:
    print(f"No se encontraron archivos .tif en la carpeta '{folder_path}'.")
    exit()

print(f"Se encontraron {len(tiff_files)} teselas.")

# --- INICIO DEL GRÁFICO DE UBICACIÓN (VERSIÓN MEJORADA) ---

# -----------------------------------------------------------------
# --- PASO: Calcular la extensión total de TODAS las teselas ---
# -----------------------------------------------------------------
print("Calculando la extensión total del mosaico...")
total_minx = float('inf')
total_miny = float('inf')
total_maxx = float('-inf')
total_maxy = float('-inf')

for file in tiff_files:
    with rasterio.open(file) as src:
        # src.bounds nos da (left, bottom, right, top)
        bounds = src.bounds
        total_minx = min(total_minx, bounds.left)
        total_miny = min(total_miny, bounds.bottom)
        total_maxx = max(total_maxx, bounds.right)
        total_maxy = max(total_maxy, bounds.top)

print(f"Límites totales calculados.")

# 3. Crear una figura y un eje para el mosaico
# Intenta con un tamaño mucho mayor, p.ej., 40x40 pulgadas
fig_mosaic, ax_mosaic = plt.subplots(1, 1, figsize=(8, 8))

# -----------------------------------------------------------------
# --- PASO: Fijar los límites del eje ---
# -----------------------------------------------------------------
ax_mosaic.set_xlim(total_minx, total_maxx)
ax_mosaic.set_ylim(total_miny, total_maxy)
# -----------------------------------------------------------------


print("Generando el mosaico con nombres y cuadrícula...")

# 4. Iterar sobre cada archivo para dibujarlo
for file in tiff_files:
    with rasterio.open(file) as src:
        # Leemos las 3 bandas (R, G, B)
        bands = src.read()

        # --- AJUSTE DE COLOR Y BRILLO ---
        max_val = 3000
        scaled_bands = np.clip(bands / max_val, 0, 1)

        # --- APILAR BANDAS PARA CREAR IMAGEN RGB ---
        rgb_image = np.dstack((scaled_bands[0], scaled_bands[1], scaled_bands[2]))

        # Usamos imshow. Ya no cambiará los límites del eje.
        ax_mosaic.imshow(rgb_image, extent=plotting_extent(src))

        # 1. DIBUJAR LA CUADRÍCULA (EL BORDE DE LA TESELA)
        bounds = src.bounds
        width = bounds.right - bounds.left
        height = bounds.top - bounds.bottom
        rect = Rectangle(
            (bounds.left, bounds.bottom), # Coordenada (x, y) de inicio
            width,                       # Ancho
            height,                      # Alto
            linewidth=1.5,               # Grosor de línea
            edgecolor='yellow',          # Color de línea (visible)
            facecolor='none',            # Sin relleno
            linestyle='--'               # Estilo de línea
        )
        ax_mosaic.add_patch(rect)
        
        # 2. AÑADIR EL NOMBRE DE LA TESELA
        center_x = bounds.left + width / 2
        center_y = bounds.bottom + height / 2
        file_name = os.path.basename(file)
        file_name = file_name[:len(file_name)-4]
        
        ax_mosaic.text(
            center_x, 
            center_y, 
            file_name, 
            color='white',         # Color del texto
            ha='center',           # Alineación horizontal
            va='center',           # Alineación vertical
            fontsize=5,            # Tamaño de fuente
            fontweight='bold',     # Negrita
            # Fondo negro semitransparente para legibilidad
            bbox=dict(facecolor='black', alpha=0.6, pad=0.2) 
        )
        # --- ¡FIN DEL NUEVO CÓDIGO! ---

# 5. Configurar y mostrar el gráfico del mosaico
ax_mosaic.set_title('Mapa de Ubicación de Teselas')
ax_mosaic.set_axis_off()

plt.show() # Muestra el primer gráfico

# --- FIN DEL GRÁFICO DE UBICACIÓN ---


# --- INICIO DEL VISUALIZADOR INDIVIDUAL ---
# (Esta parte se mantiene exactamente igual, ya funcionaba bien)

print("\n--- Visualizador de Teselas Individuales ---")

# Listar las teselas disponibles para que el usuario elija
print("Teselas encontradas:")
for i, file_path in enumerate(tiff_files):
    # Usamos os.path.basename para mostrar solo el nombre del archivo
    print(f"  {i+1}: {os.path.basename(file_path)}")

while True:
    # 1. Pedir al usuario que elija una tesela
    try:
        choice_str = input("\nIngresa el número de la tesela que quieres ver (o 'q' para salir): ")

        if choice_str.lower() == 'q':
            print("Saliendo del visualizador.")
            break # Salir del bucle while

        # Convertir la entrada a un índice de lista (base 0)
        choice_index = int(choice_str) - 1

        # 2. Validar la elección
        if 0 <= choice_index < len(tiff_files):
            file_to_show = tiff_files[choice_index]
            print(f"Cargando tesela {choice_str}: {os.path.basename(file_to_show)}...")

            # 3. Cargar y procesar la tesela elegida
            with rasterio.open(file_to_show) as src:
                bands = src.read()
                max_val = 3000
                scaled_bands = np.clip(bands / max_val, 0, 1)
                rgb_image = np.dstack((scaled_bands[0], scaled_bands[1], scaled_bands[2]))
                extent = plotting_extent(src)
                title = f"Tesela {choice_str}: {os.path.basename(file_to_show)}"

            # 4. Crear un *nuevo* gráfico para esta tesela
            fig_single, ax_single = plt.subplots(figsize=(12, 12))
            ax_single.imshow(rgb_image, extent=extent)
            ax_single.set_title(title)
            ax_single.set_axis_off()

            # 5. Mostrar el gráfico individual
            plt.show()

        else:
            print(f"Error: Número fuera de rango. Ingresa un número entre 1 y {len(tiff_files)}.")

    except ValueError:
        print("Error: Entrada no válida. Ingresa un número o 'q'.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

# --- FIN DEL VISUALIZADOR INDIVIDUAL ---