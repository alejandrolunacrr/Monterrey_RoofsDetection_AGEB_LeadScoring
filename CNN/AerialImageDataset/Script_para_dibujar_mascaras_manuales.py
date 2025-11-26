import os
import glob
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- CONFIGURACIÓN ---
INPUT_IMAGE_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\Filtros\top_20_techos_30-40_porciento\images"
OUTPUT_MASK_DIR = r"C:\Users\aleja\Desktop\google_earths\CNN\Filtros\top_20_techos_30-40_porciento\masks_MANUALES"
# ---------------------

# --- Variables Globales para manejar el estado ---
image_files = []
current_index = 0
current_image = None
current_meta = None
polygons = []
current_polygon = []
ax = None
fig = None

# --- ¡NUEVO! Variables de estado para Pan/Zoom ---
_is_panning = False
_pan_start_x = 0
_pan_start_y = 0

def load_image(index):
    """Carga y muestra la imagen en el índice dado."""
    global current_index, current_image, current_meta, polygons, current_polygon
    
    current_index = index
    path = image_files[current_index]
    
    polygons = []
    current_polygon = []
    
    with rasterio.open(path) as src:
        current_meta = src.meta.copy()
        current_image = np.transpose(src.read((1, 2, 3)), (1, 2, 0))

    ax.clear()
    ax.imshow(current_image)
    # ¡NUEVO! Restaurar los límites de la imagen (resetea el zoom)
    ax.set_xlim(0, current_image.shape[1])
    ax.set_ylim(current_image.shape[0], 0)
    
    ax.set_title(f"Imagen {current_index + 1}/{len(image_files)}: {os.path.basename(path)}")
    fig.canvas.draw()
    
    print("\n" + "="*50)
    print(f"Cargada imagen: {os.path.basename(path)}")
    print_instructions()

def draw_polygons():
    """Vuelve a dibujar todos los polígonos en el lienzo."""
    while len(ax.patches) > 0:
        ax.patches[0].remove()
    while len(ax.lines) > 0:
        ax.lines[0].remove()

    for poly_verts in polygons:
        if len(poly_verts) > 1:
            poly_widget = Polygon(poly_verts, closed=True, color='green', alpha=0.4)
            ax.add_patch(poly_widget)

    if len(current_polygon) > 0:
        xs, ys = zip(*current_polygon)
        ax.plot(xs, ys, 'ro-') 

    fig.canvas.draw()

def on_click_draw(event):
    """Manejador para clics de DIBUJO (Clic Izquierdo)."""
    global current_polygon
    
    if not event.inaxes or event.button != 1:
        return
        
    x, y = int(event.xdata), int(event.ydata)
    current_polygon.append((x, y))
    print(f"  Vértice añadido: ({x}, {y})")
    draw_polygons()

def on_key_press(event):
    """Manejador para pulsaciones de teclas."""
    global polygons, current_polygon, current_index
    
    if event.key == 'enter':
        if len(current_polygon) > 2:
            print("Polígono completado.")
            polygons.append(current_polygon)
            current_polygon = []
            draw_polygons()
        else:
            print("Se necesitan al menos 3 puntos para un polígono.")
            
    elif event.key == 'z':
        if len(current_polygon) > 0:
            removed = current_polygon.pop()
            print(f"Vértice {removed} eliminado.")
            draw_polygons()
        else:
            print("No hay vértices que deshacer.")

    # --- ¡NUEVA FUNCIÓN! ---
    elif event.key == 'b': # 'b' de "Borrar"
        if polygons: # Verificar si hay polígonos completados
            removed_poly = polygons.pop() # Eliminar el último de la lista
            print(f"Último polígono completado ({len(removed_poly)} vértices) eliminado.")
            draw_polygons()
        else:
            print("No hay polígonos completados para eliminar.")

    # --- ¡CORREGIDO! --- (Cambiado de 'l' a 'c' para coincidir con las instrucciones)
    elif event.key == '=': 
        print("¡Polígonos limpiados! Empezando de nuevo.")
        polygons = []
        current_polygon = []
        draw_polygons()

    elif event.key == 's':
        if not polygons:
            print("¡Error! No hay polígonos para guardar.")
            return
        save_mask()

    elif event.key == 'right':
        if current_index < len(image_files) - 1:
            load_image(current_index + 1)
        else:
            print("¡Ya estás en la última imagen!")

    elif event.key == 'left':
        if current_index > 0:
            load_image(current_index - 1)
        else:
            print("¡Ya estás en la primera imagen!")
            


# --- ¡NUEVAS FUNCIONES DE ZOOM Y PANEO! ---

def on_scroll_zoom(event):
    """Manejador para zoom con rueda del ratón."""
    if not event.inaxes:
        return

    base_scale = 1.1 # Factor de zoom
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    
    xdata = event.xdata # Posición X del cursor
    ydata = event.ydata # Posición Y del cursor

    if event.button == 'up': # Zoom in
        scale_factor = 1 / base_scale
    elif event.button == 'down': # Zoom out
        scale_factor = base_scale
    else:
        return

    # Calcular nuevos límites
    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    # Calcular nueva posición de los límites (centrado en el cursor)
    rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
    ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])
    fig.canvas.draw()

def on_press_pan(event):
    """Manejador para INICIAR el paneo (Clic Central)."""
    global _is_panning, _pan_start_x, _pan_start_y
    # Clic central (rueda del ratón)
    if event.button == 2:
        _is_panning = True
        _pan_start_x = event.xdata
        _pan_start_y = event.ydata

def on_release_pan(event):
    """Manejador para DETENER el paneo."""
    global _is_panning
    if event.button == 2:
        _is_panning = False

def on_motion_pan(event):
    """Manejador para MOVER la imagen durante el paneo."""
    if not _is_panning or not event.inaxes:
        return
        
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    
    # Calcular el delta (cambio)
    dx = event.xdata - _pan_start_x
    dy = event.ydata - _pan_start_y
    
    # Mover los límites del eje
    ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
    ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
    
    fig.canvas.draw()

# --- FIN DE NUEVAS FUNCIONES ---

def save_mask():
    """Crea y guarda la máscara en blanco y negro."""
    print("Guardando máscara...")
    
    mask = np.zeros(current_image.shape[:2], dtype=np.uint8)
    poly_list_int = [np.array(p, dtype=np.int32) for p in polygons]
    cv2.fillPoly(mask, poly_list_int, 255) # Rellenar con 255
    
    out_meta = current_meta.copy()
    out_meta.update({
        'driver': 'GTiff', 'dtype': 'uint8', 'count': 1, 'compress': 'lzw'
    })
    
    base_name = os.path.basename(image_files[current_index])
    save_path = os.path.join(OUTPUT_MASK_DIR, base_name)
    
    try:
        with rasterio.open(save_path, 'w', **out_meta) as dst:
            dst.write(mask, 1)
        print(f"¡Máscara guardada exitosamente en {save_path}!")
    except Exception as e:
        print(f"¡Error al guardar! {e}")

def print_instructions():
    print("\n--- ¡NUEVOS CONTROLES! ---")
    print(" Rueda del Ratón:       Zoom In/Out (centrado en el cursor).")
    print(" Clic Central + Arrastrar: Panear (mover) la imagen.")
    print("----------------------------")
    print(" Clic Izquierdo:         Añadir un vértice al polígono.")
    print(" [Enter]:               Completar el polígono actual.")
    print(" [z]:                   Deshacer (borrar) el último *vértice*.")
    print(" [b]:                   Borrar el último *polígono* completado.") # <-- ¡NUEVO!
    print(" [c]:                   Limpiar (Clear) TODOS los polígonos.") # <-- Corregido
    print(" [s]:                   Guardar (Save) la máscara en B&N.")
    print(" [→] (Flecha D):         Cargar la SIGUIENTE imagen.")
    print(" [←] (Flecha I):         Cargar la imagen ANTERIOR.")

# --- Bucle Principal de Ejecución ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    
    image_files = sorted(glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.tif")))
    
    if not image_files:
        print(f"¡Error! No se encontraron imágenes .tif en {INPUT_IMAGE_DIR}")
        exit()
        
    print(f"Encontradas {len(image_files)} imágenes para etiquetar.")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- ¡CONEXIONES DE EVENTOS ACTUALIZADAS! ---
    fig.canvas.mpl_connect('button_press_event', on_click_draw)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Conectar los nuevos eventos de pan/zoom
    fig.canvas.mpl_connect('scroll_event', on_scroll_zoom)
    fig.canvas.mpl_connect('button_press_event', on_press_pan)
    fig.canvas.mpl_connect('button_release_event', on_release_pan)
    fig.canvas.mpl_connect('motion_notify_event', on_motion_pan)
    
    load_image(0)
    
    plt.show()