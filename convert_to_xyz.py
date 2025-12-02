"""
--- СКРИПТ ДЛЯ ПРЕОБРАЗОВАНИЯ ДАТАСЕТА ОБЪЕКТОВ Wavefront OBJ В ДАТАСЕТ ПОЛНЫХ ОБЛАКОВ ТОЧЕК ФОРМАТА XYZ ---

    Загружает OBJ, "лечит" геометрию, сэмплирует, нормализует и сохраняет в XYZ

"""

import os
import glob
import trimesh
import numpy as np
# trimesh иногда выдает много предупреждений, поэтому ниже я их заглушаю
import logging
trimesh.util.attach_to_log() 

# --- КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ ---

# Папка, где лежат мои OBJ-файлы (ОБНОВИ ПУТЬ)
INPUT_FOLDER = "path_to_my_input_folder" 
# Папка, куда будут сохранены облака точек (ОБНОВИ ПУТЬ)
OUTPUT_FOLDER = "path_to_my_input_folder"
# Желаемое количество точек для сэмплирования с каждого объекта 
N_POINTS = 16384 #подобрано под стандарт PointAttN


def process_obj_to_xyz(input_path, output_path, n_points):
    
    print(f"Обработка: {os.path.basename(input_path)}...")
    
    try:
        
        mesh = trimesh.load(input_path, force='mesh', process=False) # Параметр process=False при загрузке, чтобы можно было сделать это явно позже
        
        # --- БЛОК ЛЕЧЕНИЯ ГЕОМЕТРИИ ---
        
        # Проверка: Если это Сцена (набор объектов), сливаем в одну сетку
        if isinstance(mesh, trimesh.Scene):
            print(f"  -> Обнаружена сцена (несколько объектов). Объединяем...")
            # Получаем все геометрии из сцены
            geometries = list(mesh.geometry.values())
            if len(geometries) == 0:
                print("  ОШИБКА: Пустая сцена.")
                return
            # Объединяем их в одну сетку
            mesh = trimesh.util.concatenate(geometries)

        # Базовая проверка на пустоту
        if mesh.is_empty:
            print("  ОШИБКА: Сетка пустая (нет вершин или граней).")
            return

        # Очистка и "лечение"
        mesh.process() # process=True удаляет дубликаты вершин и вырожденные грани
        
        # Г. Исправление нормалей (возможно перепутанных)
        mesh.fix_normals()
        
        
        # --- КОНВЕРТАЦИЯ ---

        # Сэмплирование точек с поверхности
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        
        # Нормализация (Центрирование и Масштабирование)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_distance = np.max(np.linalg.norm(points, axis=1))
        if max_distance > 0:
             points = points / max_distance
        
        # Сохранение
        np.savetxt(output_path, points, fmt='%.6f', delimiter=' ')
        print(f"  Успешно (точек: {len(points)}) -> {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"  КРИТИЧЕСКАЯ ОШИБКА при обработке {os.path.basename(input_path)}: {e}")


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    search_path = os.path.join(INPUT_FOLDER, "*.obj")
    obj_files = glob.glob(search_path)
    
    if not obj_files:
        print(f"В папке {INPUT_FOLDER} не найдено OBJ-файлов.")
        return
        
    print(f"Найдено {len(obj_files)} OBJ-файлов. Начало конвертации...")
    
    success_count = 0
    for obj_path in obj_files:
        base_name = os.path.basename(obj_path)
        output_name = base_name.replace(".obj", ".xyz")
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        
        try:
            process_obj_to_xyz(obj_path, output_path, N_POINTS)
            success_count += 1
        except:
            pass 

    print(f"\n✅ Готово! Успешно обработано: {success_count} из {len(obj_files)}")

if __name__ == "__main__":
    main()