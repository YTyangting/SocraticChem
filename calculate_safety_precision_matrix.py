import pandas as pd
import glob
import os

def calculate_safety_precision_matrix(folder_path):
    # 寻找所有的 _full_eval.csv 文件
    search_pattern = os.path.join(folder_path, "*_full_eval.csv")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"❌ 在 {folder_path} 下未找到评估文件")
        return

    print(f"🚀 开始计算误报率 (False Positive Analysis)...")
    print(f"{'Filename':<50} | {'FIR (误报率)':<12} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 105)

    stats_list = []

    for f_path in files:
        try:
            df = pd.read_csv(f_path)
            file_name = os.path.basename(f_path)
            
            # -----------------------------------------------------------
            # 数据准备
            # -----------------------------------------------------------
            # 我们需要 gt_status_val 和 pred_status_val
            # 0 = FAIL (Hazard / Intercept) --> 正类 (Positive)
            # 1 = PASS (Safe / Proceed)     --> 负类 (Negative)
            
            # 过滤掉无效数据 (-1)
            valid_df = df[(df['gt_status_val'].isin([0, 1])) & (df['pred_status_val'].isin([0, 1]))].copy()
            
            if len(valid_df) == 0:
                continue

            # -----------------------------------------------------------
            # 混淆矩阵计算 (Confusion Matrix)
            # -----------------------------------------------------------
            # TP (True Positive):  GT=Hazard(0) & Pred=Hazard(0) -> 正确拦截
            # FN (False Negative): GT=Hazard(0) & Pred=Safe(1)   -> 漏报 (危险!)
            # FP (False Positive): GT=Safe(1)   & Pred=Hazard(0) -> 误报 (扰民)
            # TN (True Negative):  GT=Safe(1)   & Pred=Safe(1)   -> 正确放行

            TP = ((valid_df['gt_status_val'] == 0) & (valid_df['pred_status_val'] == 0)).sum()
            FN = ((valid_df['gt_status_val'] == 0) & (valid_df['pred_status_val'] == 1)).sum()
            FP = ((valid_df['gt_status_val'] == 1) & (valid_df['pred_status_val'] == 0)).sum()
            TN = ((valid_df['gt_status_val'] == 1) & (valid_df['pred_status_val'] == 1)).sum()

            # -----------------------------------------------------------
            # 指标计算
            # -----------------------------------------------------------
            # 1. Recall (Intercept Rate) = TP / (TP + FN)
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            
            # 2. False Intercept Rate (FIR) = FP / (FP + TN)
            # 定义：在所有本来安全(GT=Safe)的样本中，有多少被错误拦截了？
            fir = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            
            # 3. Precision = TP / (TP + FP)
            # 定义：在所有模型认为危险的样本中，有多少是真的危险？
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            
            # 4. F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # -----------------------------------------------------------
            # 输出 & 记录
            # -----------------------------------------------------------
            print(f"{file_name:<50} | {fir:.2%}       | {precision:.2%}     | {recall:.2%}     | {f1:.2f}")
            
            stats_list.append({
                'Model': file_name.replace('predictions_', '').replace('_new_xml_full_eval.csv', '').replace('_full_eval.csv', ''),
                'FIR': fir,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN
            })

        except Exception as e:
            print(f"❌ Error: {e}")

    # 保存统计结果到 CSV，方便画图
    if stats_list:
        out_df = pd.DataFrame(stats_list)
        out_df.to_csv("safety_precision_analysis.csv", index=False)
        print(f"\n💾 详细数据已保存至 'safety_precision_analysis.csv'")

# 使用示例
target_folder = "/home/yjh/socChemlab/new_test" 
calculate_safety_precision_matrix(target_folder)