import os
import glob
import rasterio
from rasterio.plot import plotting_extent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import shutil
from matplotlib.widgets import Button

# 1. Definir la ruta a la carpeta que contiene las teselas
folder_path = r'C:\Users\aleja\Desktop\google_earths\manejo_imagenes_geotiff\imagenes_divididas2\cuadrante_4'
# Definir la carpeta de salida para la selección
output_folder = 'seleccion'

# --- Variables Globales para la interactividad ---
selected_files = set()  # Un 'set' para guardar las rutas de los archivos seleccionados
tile_data = []          # Lista para guardar los datos de cada tesela (archivo, límites, patch)
click_event_id = None   # Para poder desconectar el evento de clic durante la verificación

# Estilos visuales para la selección
STYLE_DESELECTED = {
    'linewidth': 1.0,
    'edgecolor': 'white',
    'facecolor': 'none',
    'linestyle': ':',
    'alpha': 0.4
}
STYLE_SELECTED = {
    'linewidth': 2.0,
    'edgecolor': 'yellow',
    'facecolor': (1, 1, 0, 0.3), # Amarillo semitransparente
    'linestyle': '-',
    'alpha': 0.3
}
# --------------------------------------------------

# Verificar si la carpeta existe
if not os.path.isdir(folder_path):
    print(f"Error: La carpeta '{folder_path}' no fue encontrada.")
    exit()

# 2. Encontrar todos los archivos GeoTIFF (.tif)
search_criteria = "*.tif"
tiff_files = glob.glob(os.path.join(folder_path, search_criteria))
tiff_files.sort() 

if not tiff_files:
    print(f"No se encontraron archivos .tif en la carpeta '{folder_path}'.")
    exit()

print(f"Se encontraron {len(tiff_files)} teselas. Haz clic para seleccionar.")

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
        bounds = src.bounds
        total_minx = min(total_minx, bounds.left)
        total_miny = min(total_miny, bounds.bottom)
        total_maxx = max(total_maxx, bounds.right)
        total_maxy = max(total_maxy, bounds.top)

print(f"Límites totales calculados.")

# 3. Crear una figura y un eje para el mosaico
fig_mosaic, ax_mosaic = plt.subplots(1, 1, figsize=(10, 10))
plt.subplots_adjust(left=0.02, bottom=0.1, right=0.98, top=0.95)
ax_mosaic.set_xlim(total_minx, total_maxx)
ax_mosaic.set_ylim(total_miny, total_maxy)

print("Generando el mosaico interactivo...")

# 4. Iterar sobre cada archivo para dibujarlo
for file in tiff_files:
    with rasterio.open(file) as src:
        bands = src.read()
        max_val = 3000
        scaled_bands = np.clip(bands / max_val, 0, 1)
        rgb_image = np.dstack((scaled_bands[0], scaled_bands[1], scaled_bands[2]))

        ax_mosaic.imshow(rgb_image, extent=plotting_extent(src))

        bounds = src.bounds
        width = bounds.right - bounds.left
        height = bounds.top - bounds.bottom
        
        rect = Rectangle(
            (bounds.left, bounds.bottom), 
            width, 
            height, 
            **STYLE_DESELECTED 
        )
        ax_mosaic.add_patch(rect)
        
        center_x = bounds.left + width / 2
        center_y = bounds.bottom + height / 2
        file_name = os.path.basename(file)
        file_name = file_name[:len(file_name)-4]
        
        ax_mosaic.text(
            center_x, 
            center_y, 
            file_name, 
            color='white', 
            ha='center', 
            va='center', 
            fontsize=5, 
            fontweight='bold', 
            bbox=dict(facecolor='black', alpha=0.6, pad=0.2) 
        )
        
        tile_data.append({
            'file': file,
            'bounds': bounds,
            'patch': rect  
        })

ax_mosaic.set_title('Mapa de Ubicación de Teselas')
ax_mosaic.set_axis_off()


# --- DEFINICIÓN DE FUNCIONES INTERACTIVAS ---

def on_click(event):
    """Manejador de eventos para el clic del ratón EN EL MAPA."""
    if event.inaxes != ax_mosaic:
        return

    x, y = event.xdata, event.ydata
    
    for tile in tile_data:
        bounds = tile['bounds']
        
        if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
            file_path = tile['file']
            patch = tile['patch']
            
            if file_path in selected_files:
                selected_files.remove(file_path)
                patch.set(**STYLE_DESELECTED)
                print(f"Deseleccionado: {os.path.basename(file_path)}")
            else:
                selected_files.add(file_path)
                patch.set(**STYLE_SELECTED)
                print(f"Seleccionado: {os.path.basename(file_path)}")
            
            fig_mosaic.canvas.draw()
            break 

# --- ¡NUEVA CLASE PARA LA VERIFICACIÓN! ---
class VerificationApp:
    def __init__(self, files_set_to_verify):
        self.files_to_verify = list(files_set_to_verify) # Lista de archivos a revisar
        self.final_selection = set(files_set_to_verify) # Set que modificaremos
        self.current_index = 0

        # Crear la nueva figura para la verificación
        self.fig, self.ax = plt.subplots(figsize=(10, 11))
        self.fig.subplots_adjust(bottom=0.2)
        
        # Crear botones
        ax_keep = plt.axes([0.25, 0.05, 0.2, 0.075])
        self.btn_keep = Button(ax_keep, 'Conservar')
        self.btn_keep.on_clicked(self.on_keep)

        ax_discard = plt.axes([0.55, 0.05, 0.2, 0.075])
        self.btn_discard = Button(ax_discard, 'Descartar')
        self.btn_discard.on_clicked(self.on_discard)

        # Mostrar la primera imagen
        self.show_next_image()
        plt.show() # Bloquea hasta que esta ventana se cierre

    def show_next_image(self):
        """Muestra la siguiente imagen de la lista o termina."""
        
        # Si ya no hay imágenes, terminar el proceso
        if self.current_index >= len(self.files_to_verify):
            self.on_finish()
            return

        file_path = self.files_to_verify[self.current_index]
        
        # Cargar y procesar la imagen
        with rasterio.open(file_path) as src:
            bands = src.read()
            max_val = 3000
            scaled_bands = np.clip(bands / max_val, 0, 1)
            rgb_image = np.dstack((scaled_bands[0], scaled_bands[1], scaled_bands[2]))
            extent = plotting_extent(src)
            title = f"Verificando {self.current_index + 1}/{len(self.files_to_verify)}: {os.path.basename(file_path)}"
        
        # Dibujar la imagen en el eje
        self.ax.clear()
        self.ax.imshow(rgb_image, extent=extent)
        self.ax.set_title(title)
        self.ax.set_axis_off()
        self.fig.canvas.draw()

    def on_keep(self, event):
        """Conserva la imagen actual y pasa a la siguiente."""
        print(f"Conservado: {os.path.basename(self.files_to_verify[self.current_index])}")
        self.current_index += 1
        self.show_next_image()

    def on_discard(self, event):
        """Descarta la imagen actual, la quita de la selección y pasa a la siguiente."""
        file_path = self.files_to_verify[self.current_index]
        
        print(f"Descartado: {os.path.basename(file_path)}")
        
        # 1. Quitar del set de selección final
        self.final_selection.remove(file_path)
        
        # 2. Actualizar el set global (para el mapa)
        global selected_files
        selected_files.remove(file_path)

        # 3. Actualizar visualmente el mapa principal
        for tile in tile_data:
            if tile['file'] == file_path:
                tile['patch'].set(**STYLE_DESELECTED)
                fig_mosaic.canvas.draw()
                break
        
        # 4. Pasar a la siguiente imagen
        self.current_index += 1
        self.show_next_image()

    def on_finish(self):
        """Se llama cuando se han revisado todas las imágenes."""
        print("\n--- Verificación Terminada ---")
        plt.close(self.fig) # Cierra la ventana de verificación

        # Reactivar los clics en el mapa principal
        global click_event_id
        click_event_id = fig_mosaic.canvas.mpl_connect('button_press_event', on_click)
        
        if not self.final_selection:
            print("No quedó ninguna tesela en la selección final.")
            return

        # Preguntar por la exportación final en la consola
        print(f"Quedan {len(self.final_selection)} teselas en la selección final.")
        try:
            choice = input("¿Deseas exportar estas teselas ahora? [s/n]: ").lower()
            if choice == 's':
                export_final_selection(self.final_selection)
            else:
                print("Exportación cancelada. Puedes seguir seleccionando.")
        except EOFError:
            print("Entrada no detectada. Exportación cancelada.")

# --- Funciones de Botones Principales ---

def start_verification(event):
    """Callback del botón 'Verificar'. Inicia la app de verificación."""
    
    if not selected_files:
        print("\nNo hay archivos seleccionados para verificar.")
        return

    print(f"\nIniciando verificación de {len(selected_files)} teselas...")
    
    # Desconectar clics del mapa principal para evitar conflictos
    global click_event_id
    if click_event_id:
        fig_mosaic.canvas.mpl_disconnect(click_event_id)
        click_event_id = None # Asegurarse que está desconectado
    
    # Crear la instancia de la app de verificación
    # El script se pausará aquí hasta que la ventana de verificación se cierre
    verification_app = VerificationApp(selected_files)


def export_final_selection(final_files_set):
    """Función final que copia los archivos verificados."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\nExportando {len(final_files_set)} archivos a la carpeta '{output_folder}'...")
    
    count = 0
    for file_path in final_files_set:
        try:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(output_folder, filename)
            shutil.copy(file_path, dest_path)
            count += 1
        except Exception as e:
            print(f"Error al copiar {filename}: {e}")
            
    print(f"¡Éxito! Se copiaron {count} archivos.")


# --- CONECTAR EVENTOS Y MOSTRAR GRÁFICO ---

# Conectar el evento de clic a la función on_click
click_event_id = fig_mosaic.canvas.mpl_connect('button_press_event', on_click)

# Añadir el botón principal (ahora de verificación)
ax_button_verify = plt.axes([0.4, 0.015, 0.2, 0.075])
btn_verify = Button(ax_button_verify, 'Verificar Selección')
btn_verify.on_clicked(start_verification) # Llama a la nueva función


plt.show()

print("Aplicación cerrada.")