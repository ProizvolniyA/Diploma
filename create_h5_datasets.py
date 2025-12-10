"""
--- СКРИПТ ДЛЯ СОСТАВЛЕНИЯ ПАРНОГО ДАТАСЕТА <PARTICIAL, GT> ИЗ ДАТАСЕТОВ ОБЪЕКТОВ И ИХ ОСКОЛКОВ---

    Сопоставляет полные облака точек восстановленной керамики (Ground_True) соответствующим неполным облакам точек 
    полученных из них осколков (Partial), помещает полученныt пары в контейнеры h5 и систематически делит датасет 
    на обучающий и тестовый наборы данных.

"""
import os
import glob
import numpy as np
import h5py

# --- КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ ---

PARTIAL_FOLDER = r"path_to_my_partial_clouds"      # Папка с осколками (вход сети)
COMPLETE_FOLDER = r"path_to_my_complete_clouds"    # Папка с полными облаками (выход/GT)
OUTPUT_FOLDER = r"path_to_my_output_folder"        # Папка для полученных .h5 файлов    
# Если поставить 6, то каждый 6-й файл уйдет в тест (примерно 16.6% данных)
# Если поставить 5, то каждый 5-й (20% данных)
# Если поставить 10, то каждый 10-й (10% данных)
TEST_STEP = 6 

def load_xyz(path):
    try:
        return np.loadtxt(path, dtype=np.float32)
    except Exception as e:
        print(f"Ошибка чтения {path}: {e}")
        return None

def create_h5_file(output_path, partial_list, complete_list):
    partial_np = np.stack(partial_list, axis=0)
    complete_np = np.stack(complete_list, axis=0)
    
    print(f"  Запись в {os.path.basename(output_path)}...")
    print(f"    Размерность входа (partial): {partial_np.shape}")
    print(f"    Размерность выхода (gt): {complete_np.shape}")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset("partial", data=partial_np)
        f.create_dataset("gt", data=complete_np)

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. Находим файлы и ОБЯЗАТЕЛЬНО сортируем их
    # sorted() гарантирует, что файлы идут по порядку имен
    partial_files = sorted(glob.glob(os.path.join(PARTIAL_FOLDER, "*.xyz")))
    complete_files = sorted(glob.glob(os.path.join(COMPLETE_FOLDER, "*.xyz")))
    
    complete_map = {os.path.basename(p): p for p in complete_files}
    
    data_pairs = [] 
    
    print("Чтение и сопоставление файлов...")
    for p_path in partial_files:
        name = os.path.basename(p_path)
        
        if name in complete_map:
            c_path = complete_map[name]
            p_cloud = load_xyz(p_path)
            c_cloud = load_xyz(c_path)
            
            if p_cloud is not None and c_cloud is not None:
                data_pairs.append((p_cloud, c_cloud))
        else:
            print(f"ВНИМАНИЕ: Не найдена пара для {name}")

    total_items = len(data_pairs)
    print(f"Всего валидных пар: {total_items}")
    
    if total_items == 0: return

    # 2. Разделение на Train и Test с использованием СИСТЕМАТИЧЕСКОЙ ВЫБОРКИ
    # в зависимости от датасета вместо шага можно сделать random.shuffle, в моём случае шаг будет лучше
    
    train_pairs = []
    test_pairs = []
    
    for i, pair in enumerate(data_pairs):
        if (i + 1) % TEST_STEP == 0:
            test_pairs.append(pair)
        else:
            train_pairs.append(pair)
            
    print(f"\nРазделение завершено (Шаг = {TEST_STEP}):")
    print(f"  Обучающая выборка: {len(train_pairs)} файлов")
    print(f"  Тестовая выборка:  {len(test_pairs)} файлов")
    
    # 3. Подготовка списков
    train_partial = [p[0] for p in train_pairs]
    train_complete = [p[1] for p in train_pairs]
    
    test_partial = [p[0] for p in test_pairs]
    test_complete = [p[1] for p in test_pairs]
    
    # 4. Сохранение
    create_h5_file(os.path.join(OUTPUT_FOLDER, "train.h5"), train_partial, train_complete)
    create_h5_file(os.path.join(OUTPUT_FOLDER, "test.h5"), test_partial, test_complete)
    
    print("\n✅ Обучающий и тестовый датасеты (H5) успешно созданы!")

if __name__ == "__main__":
    main()
   