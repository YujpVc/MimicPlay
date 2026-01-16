import numpy as np
import math

def calculate_camera_parameters():
    """
    Calcula y muestra los parámetros intrínsecos y extrínsecos para la cámara
    'agentview_image' definida en la clase GenesisEnvWrapper.
    """
    
    # --- Parámetros de Entrada (extraídos del código de la clase) ---
    # Resolución por defecto de la cámara
    width = 84
    height = 84
    
    # Parámetros extrínsecos definidos en self.scene.add_camera()
    camera_pos = np.array([2.5, 1.0, 1.8])
    camera_lookat = np.array([0.65, 1.0, 1.0])
    
    # Parámetro intrínseco (Campo de Visión Vertical) en grados
    fov_vertical_deg = 30.0
    
    # Se asume un vector "arriba" estándar para el mundo (convención común Y-arriba)
    world_up = np.array([0.0, 1.0, 0.0])

    print("="*50)
    print("Análisis de Parámetros de la Cámara 'agentview_image'")
    print("="*50)

    # --- 1. Parámetros Extrínsecos ---
    print("\n--- 1. Parámetros Extrínsecos (Pose de la Cámara) ---\n")
    print(f"Posición de la Cámara (t): {camera_pos}")
    print(f"Punto de Mira de la Cámara: {camera_lookat}")
    print(f"Vector 'Arriba' del Mundo asumido: {world_up}\n")

    # Calcular la matriz de vista (View Matrix) [R | t]
    # Esta matriz transforma los puntos del espacio del mundo al espacio de la cámara.
    
    # El eje z de la cámara apunta desde el punto de mira a la cámara
    z_axis = camera_pos - camera_lookat
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # El eje x de la cámara es perpendicular al vector "arriba" del mundo y al eje z
    x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # El eje y de la cámara es perpendicular a los ejes z y x
    y_axis = np.cross(z_axis, x_axis)

    # La parte de rotación de la matriz de vista es la transpuesta de la matriz 
    # de base formada por los ejes de la cámara.
    rotation_matrix = np.array([x_axis, y_axis, z_axis])
    
    # La parte de traslación se calcula como -R * t
    translation_vector = -np.dot(rotation_matrix, camera_pos)
    
    # La matriz extrínseca completa (Matriz de Vista)
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
    
    print("Matriz Extrínseca (Matriz de Vista):")
    print("Transforma coordenadas del mundo a coordenadas de la cámara.")
    print(np.round(extrinsic_matrix, 4))
    
    
    # --- 2. Parámetros Intrínsecos ---
    print("\n--- 2. Parámetros Intrínsecos (Propiedades de la Cámara) ---\n")
    print(f"Resolución: {width}x{height} píxeles")
    print(f"Campo de Visión Vertical (FOV): {fov_vertical_deg} grados\n")

    # Punto Principal (cx, cy): centro óptico de la imagen.
    # Se asume que está en el centro de la imagen.
    cx = width / 2.0
    cy = height / 2.0

    # Distancia Focal (fx, fy) en píxeles.
    # Se calcula a partir del campo de visión vertical (fov_y).
    fov_vertical_rad = math.radians(fov_vertical_deg)
    fy = height / (2 * math.tan(fov_vertical_rad / 2))
    
    # Se asumen píxeles cuadrados, por lo que fx = fy.
    fx = fy

    # Matriz Intrínseca (K)
    # K = [[fx,  0, cx],
    #      [ 0, fy, cy],
    #      [ 0,  0,  1]]
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0,  1]
    ])

    print(f"Punto Principal (cx, cy): ({cx}, {cy})")
    print(f"Distancia Focal en Píxeles (fx, fy): ({fx:.4f}, {fy:.4f})\n")
    print("Matriz Intrínseca (K):")
    print(np.round(intrinsic_matrix, 4))
    print("="*50)

if __name__ == '__main__':
    calculate_camera_parameters()

