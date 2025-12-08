"""
--- СКРИПТ ДЛЯ ГЕНЕРАЦИИ ДАТАСЕТА РАЗРЯЖЁННЫХ НЕПОЛНЫХ ОБЛАКОВ ТОЧЕК ИЗ ДАТАСЕТА ПОЛНЫХ ОБЛАКОВ ТОЧЕК ФОРМАТА XYZ ---

    Генерирует случайные "осколки" или фрагменты, соответствующие повреждённой керамике
    В данном случае рассмотрен вариант генерации одного цельного "осколка", разряжённого до облака из 2048 точек

"""

import os
import glob
import numpy as np

# --- КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ ---

INPUT_FOLDER = "path_to_my_input_folder"   # Папка с полными облаками (16384) (ОБНОВИ ПУТЬ)
OUTPUT_FOLDER = "path_to_my_output_folder" # Папка для осколков (2048) (ОБНОВИ ПУТЬ)
CROP_SIZE = 5120   # Промежуточный размер 
FINAL_SIZE = 2048  # Итоговый размер 


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

"""Создает связный фрагмент, выбирая случайную точку и её соседей, а затем разряжает его."""
def generate_sparse_fragment(points, crop_size, final_size):
    
    total_points = points.shape[0]
    
    # Защита, если в файле меньше точек, чем мы хотим вырезать
    if total_points < crop_size:
        print(f"  Внимание: точек мало ({total_points}), берем все.")
        crop = points
    else:
        # 1.KNN: 
        # Выбор случайного индекса "центра" осколка
        center_idx = np.random.randint(0, total_points)
        center_point = points[center_idx]
        
        # Вычисление евклидовых расстояний от центра до всех остальных точек
        distances = np.linalg.norm(points - center_point, axis=1)
        
        # Сортировка ближайших точек по расстоянию
        nearest_indices = np.argsort(distances)[:crop_size] #возвращает индексы отсортированного массива
        crop = points[nearest_indices]

    # 2.Этап Прореживания (Subsampling): 
    # Случайный выбор точек из куска без повторений
    current_crop_size = crop.shape[0]
    
    if current_crop_size <= final_size:
        return crop
    
    random_indices = np.random.choice(current_crop_size, final_size, replace=False)
    final_fragment = crop[random_indices]
    
    return final_fragment

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    search_path = os.path.join(INPUT_FOLDER, "*.xyz")
    files = glob.glob(search_path)
    
    if not files:
        print("Файлы не найдены.")
        return

    print(f"Найдено {len(files)} файлов. Начало генерации...")

    for i, file_path in enumerate(files):
        points = load_xyz(file_path)
        if points is None: continue

        fragment = generate_sparse_fragment(points, CROP_SIZE, FINAL_SIZE)

        base_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, base_name)
        save_xyz(output_path, fragment)

        if (i+1) % 100 == 0:
             print(f"Обработано {i+1} / {len(files)}...")

    print("✅ Готово! Датасет разрушенных артефактов создан.")

if __name__ == "__main__":
    main()
