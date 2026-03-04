import os
import shutil
import random

def split_dataset_by_experiment(source_folder, output_folder, test_ratio=0.1):
    """
    按“实验种类”划分数据集。
    测试集中的实验名称将不会出现在训练集中。
    """
    # 1. 准备路径
    train_dir = os.path.join(output_folder, 'train')
    test_dir = os.path.join(output_folder, 'test')
    
    if os.path.exists(output_folder):
        print(f"警告: 输出目录 '{output_folder}' 已存在，建议先清空或使用新目录以免混淆。")
    else:
        os.makedirs(output_folder)
        
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    if not os.path.exists(test_dir): os.makedirs(test_dir)

    # 2. 扫描并按照实验名称分组
    all_files = [f for f in os.listdir(source_folder) if f.endswith('.jsonl')]
    experiment_map = {} # 格式: {'实验名': [文件1, 文件2, ...]}

    for filename in all_files:
        # 提取实验名称 (key)
        # 根据您图片的格式，key 是 "1_大理石与稀盐酸反应" 这一段
        try:
            exp_key = filename.split('_sample')[0]
        except IndexError:
            exp_key = "unknown"
            
        if exp_key not in experiment_map:
            experiment_map[exp_key] = []
        experiment_map[exp_key].append(filename)

    # 3. 获取所有唯一的实验名称列表
    unique_experiments = list(experiment_map.keys())
    total_exps = len(unique_experiments)
    
    print(f"扫描完成：共发现 {total_exps} 种不同的化学实验。")
    if total_exps < 2:
        print("【错误】实验种类少于2种，无法进行实验级划分（至少需要一种进训练集，一种进测试集）。")
        return

    # 4. 随机抽取实验进入测试集
    random.shuffle(unique_experiments)
    
    # 计算测试集需要包含多少个实验
    num_test_exps = int(total_exps * test_ratio)
    # 保证至少有一个实验在测试集（除非总数太少）
    if num_test_exps == 0 and total_exps > 1:
        num_test_exps = 1
        
    test_experiment_names = unique_experiments[:num_test_exps]
    train_experiment_names = unique_experiments[num_test_exps:]

    print(f"划分结果：{len(train_experiment_names)} 种实验进入训练集，{len(test_experiment_names)} 种实验进入测试集。")
    print(f"测试集包含的实验: {test_experiment_names}")

    # 5. 执行文件复制/移动
    count_train_files = 0
    count_test_files = 0

    # 处理训练集
    for exp_name in train_experiment_names:
        files = experiment_map[exp_name]
        for f in files:
            shutil.copy2(os.path.join(source_folder, f), os.path.join(train_dir, f))
        count_train_files += len(files)

    # 处理测试集
    for exp_name in test_experiment_names:
        files = experiment_map[exp_name]
        for f in files:
            shutil.copy2(os.path.join(source_folder, f), os.path.join(test_dir, f))
        count_test_files += len(files)

    print("-" * 30)
    print(f"完成！")
    print(f"Train 文件夹: {count_train_files} 个文件 (涵盖 {len(train_experiment_names)} 种实验)")
    print(f"Test  文件夹: {count_test_files} 个文件 (涵盖 {len(test_experiment_names)} 种实验)")

# --- 运行配置 ---
if __name__ == "__main__":
    SOURCE_DIR = 'finetune_data_dataset' 
    OUTPUT_DIR = 'dataset_experiment_split' # 输出文件夹名字改了一下，区分上一次的
    
    if os.path.exists(SOURCE_DIR):
        split_dataset_by_experiment(SOURCE_DIR, OUTPUT_DIR, test_ratio=0.1)
    else:
        print(f"找不到源文件夹: {SOURCE_DIR}")