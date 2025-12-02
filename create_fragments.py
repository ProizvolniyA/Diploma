"""
--- СКРИПТ ДЛЯ ГЕНЕРАЦИИ ДАТАСЕТА РАЗРЯЖЁННЫХ НЕПОЛНЫХ ОБЛАКОВ ТОЧЕК ИЗ ДАТАСЕТА ПОЛНЫХ ОБЛАКОВ ТОЧЕК ФОРМАТА XYZ ---

    Генерирует случайные "осколки" или фрагменты, соответствующие повреждённой керамике

"""

import os
import glob
import numpy as np

# --- КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ ---

# Папка с полными облаками (16384)
INPUT_FOLDER = "D:\Armen\All Clouds"  
# Папка для осколков (2048)
OUTPUT_FOLDER = "D:\Armen\Fragments" 
TARGET_POINTS = 2048  # Сколько точек оставить в "осколке" (постандарту PointAttN)


"""Загружает XYZ файл в массив numpy."""
def load_xyz(path):
    
    try:
        return np.loadtxt(path, dtype=np.float32)
    except Exception as e:
        print(f"Ошибка чтения {path}: {e}")
        return None
    
"""Сохраняет массив numpy в XYZ файл."""
def save_xyz(path, points):
    
    np.savetxt(path, points, fmt='%.6f', delimiter=' ')
    
    
"""Создает связный фрагмент, выбирая случайную точку и её соседей."""
def generate_fragment(points, num_points):
    
    total_points = points.shape[0]
    
    if total_points < num_points:
        return points

    # Выбор случайного индекса "центра" осколка
    center_idx = np.random.randint(0, total_points)
    center_point = points[center_idx]

    # Вычисление евклидовых расстояний от центра до всех остальных точек
    distances = np.linalg.norm(points - center_point, axis=1)

    # Сортировка ближайших точек по расстоянию
    
    nearest_indices = np.argsort(distances)[:num_points] #возвращает индексы отсортированного массива

    fragment = points[nearest_indices]
    
    return fragment

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Создана папка: {OUTPUT_FOLDER}")

    search_path = os.path.join(INPUT_FOLDER, "*.xyz")
    files = glob.glob(search_path)
    
    if not files:
        print("Файлы не найдены.")
        return

    print(f"Найдено {len(files)} файлов. Начало фрагментации...")

    for i, file_path in enumerate(files):
        
        points = load_xyz(file_path)
        if points is None: continue

        fragment = generate_fragment(points, TARGET_POINTS)

        base_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, base_name)
        save_xyz(output_path, fragment)

        if (i+1) % 100 == 0:
             print(f"Обработано {i+1} / {len(files)}...")

    print("✅ Готово! Датасет разрушенных артефактов создан.")

if __name__ == "__main__":
    main()