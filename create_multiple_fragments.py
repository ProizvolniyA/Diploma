"""
--- СКРИПТ ДЛЯ ГЕНЕРАЦИИ ДАТАСЕТА РАЗРЯЖЁННЫХ НЕПОЛНЫХ ОБЛАКОВ ТОЧЕК ИЗ ДАТАСЕТА ПОЛНЫХ ОБЛАКОВ ТОЧЕК ФОРМАТА XYZ ---

    Генерирует случайные "осколки" или фрагменты, соответствующие повреждённой керамике
    В данном случае рассмотрен вариант генерации двух цельных "осколков" одного артефакта, чьё общее облако разряжено до 2048 точек

"""
import os
import glob
import numpy as np

# --- КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ ---

INPUT_FOLDER = "path_to_my_input_folder"   # Папка с полными облаками (16384) (ОБНОВИ ПУТЬ)
OUTPUT_FOLDER = "path_to_my_output_folder" # Папка для осколков (2048) (ОБНОВИ ПУТЬ)
FRAGMENT_SIZE = 3072   # Размер каждого отдельного куска
FINAL_SIZE = 2048      # Итоговый размер выходного облака

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

def get_knn_indices(points, center_idx, k):
    """Возвращает индексы k ближайших точек к центру."""
    center_point = points[center_idx]
    distances = np.linalg.norm(points - center_point, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices

""" Генерирует два фрагмента и объединяет их с прореживанием. """
def generate_multipart_fragment(points, frag_size, final_size):
    
    n_points = points.shape[0]
    all_indices = np.arange(n_points)
    
    if n_points <= frag_size:
        return points

    # --- ФРАГМЕНТ 1 ---
    seed1 = np.random.choice(all_indices)
    indices1 = get_knn_indices(points, seed1, frag_size)
    
    # --- ФРАГМЕНТ 2 ---
    # Исключение точек первого фрагмента из кандидатов на центр второго, чтобы увеличить шанс, что фрагменты будут в разных местах.
    
    available_seeds = np.setdiff1d(all_indices, indices1) #возвращает элементы из all_indices, которых нет в indices1
    
    if len(available_seeds) > 0:
        seed2 = np.random.choice(available_seeds)
    else:
        # Если вдруг первый фрагмент покрыл всё облако (но это очень маловероятно)
        seed2 = np.random.choice(all_indices)
        
    indices2 = get_knn_indices(points, seed2, frag_size)
    
    # --- ОБЪЕДИНЕНИЕ (UNION) ---
    
    combined_indices = np.union1d(indices1, indices2)# автоматически убирает дубликаты, если фрагменты пересеклись
    combined_points = points[combined_indices]
    
    # --- ПРОРЕЖИВАНИЕ (DOWNSAMPLING) ---
    current_size = combined_points.shape[0]
    
    if current_size <= final_size:
        return combined_points
    
    # Случайная выборка точек осколков без повторений
    random_subset_indices = np.random.choice(current_size, final_size, replace=False)
    final_fragment = combined_points[random_subset_indices]
    
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

        fragment = generate_multipart_fragment(points, FRAGMENT_SIZE, FINAL_SIZE)

        base_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, base_name)
        save_xyz(output_path, fragment)

        if (i+1) % 100 == 0:
             print(f"Обработано {i+1} / {len(files)}...")

    print("✅ Готово! Датасет с двойными фрагментами создан.")

if __name__ == "__main__":

    main()
