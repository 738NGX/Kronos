"""
通用测试工具函数
"""
import os
import sys
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def setup_environment(seed: int = 100):
    """
    配置测试环境（字体、路径、随机数种子）
    
    Args:
        seed: int, 随机数种子（默认100，与config.py保持一致）
    """
    # ================= 随机数种子设置 =================
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # ================= 环境配置 =================
    # 设置中文字体
    plt.rcParams['font.family'] = 'Noto Serif CJK JP'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 确保模型路径可访问
    sys.path.append("/gemini/code/")


def save_prediction_results(predictions, output_dir, model_name, index_name):
    """
    保存单个指数的预测结果
    
    Args:
        predictions: list, 预测结果列表
        output_dir: str, 输出目录
        model_name: str, 模型名称 (base/finetuned)
        index_name: str, 指数名称
    
    Returns:
        pd.DataFrame: 预测结果DataFrame
    """
    res_df = pd.DataFrame(predictions)
    output_path = os.path.join(output_dir, f"predictions_{model_name}_{index_name}.csv")
    res_df.to_csv(output_path, index=False)
    return res_df


def aggregate_and_save_metrics(all_metrics, output_dir, model_name):
    """
    汇总所有指标并保存
    
    Args:
        all_metrics: list, 所有指数的指标列表
        output_dir: str, 输出目录
        model_name: str, 模型名称
    
    Returns:
        pd.DataFrame: 汇总后的指标DataFrame
    """
    from testutils.metrics_utils import save_and_print_metrics
    
    if all_metrics:
        final_df = pd.concat(all_metrics, ignore_index=True)
        save_and_print_metrics(final_df, output_dir, model_name=model_name)
        return final_df
    return None


def plot_all_results(all_results, output_dir, model_name, test_config, combine_subplots=True):
    """
    绘制所有指数的预测曲线
    
    Args:
        all_results: dict, 所有指数的预测结果
        output_dir: str, 输出目录
        model_name: str, 模型名称
        test_config: dict, 测试配置
        combine_subplots: bool, 是否组合为大图
    """
    from testutils.visualization_utils import plot_predictions
    
    if all_results:
        print("\n🎨 开始绘制预测曲线...")
        plot_predictions(all_results, output_dir, model_name=model_name, 
                        test_config=test_config, combine_subplots=combine_subplots)


def parse_test_args(description):
    """
    解析测试脚本的命令行参数
    
    Args:
        description: str, 脚本描述
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--separate-plots', action='store_true', 
                       help='输出独立图表而非组合大图')
    return parser.parse_args()


def run_batch_inference(
    predictor, 
    all_data, 
    indices_dict, 
    config, 
    output_dir,
    model_name,
    preprocess_fn=None,
    denormalize_fn=None
):
    """
    批量推理优化版本 - 支持并行预测多个指数
    
    Args:
        predictor: KronosPredictor instance
        all_data: 所有指数的原始数据
        indices_dict: 指数名称->代码 的字典
        config: 配置字典，包含 lookback, pred_len, test_start, test_end, T, top_p, sample_count 等
        output_dir: 输出目录
        model_name: 模型名称 (base/finetuned)
        preprocess_fn: 预处理函数（仅 finetuned 需要）
        denormalize_fn: 反归一化函数（仅 finetuned 需要）
    
    Returns:
        tuple: (all_metrics, all_results)
    """
    from testutils.data_utils import load_and_prepare_index_data
    from testutils.metrics_utils import calculate_metrics
    
    # 1. 为所有指数加载并预处理数据
    print("📥 加载所有指数数据...")
    indices_data = {}
    test_indices_dict = {}
    
    for name, symbol in indices_dict.items():
        df, test_indices = load_and_prepare_index_data(all_data, name, symbol, config)
        if df is not None:
            indices_data[name] = df
            test_indices_dict[name] = test_indices
        else:
            print(f"⚠️ Skipping {name}: 数据加载失败")
    
    if not indices_data:
        print("❌ No valid index data found!")
        return [], {}
    
    # 2. 获取所有有效指数的测试索引
    # 需要找到所有指数都有数据的索引子集（用于批量预测）
    all_indices_set = [set(test_indices_dict[name]) for name in indices_data.keys()]
    common_indices = sorted(list(set.intersection(*all_indices_set)))
    
    if not common_indices:
        print("❌ No common test indices across all valid indices!")
        return [], {}
    
    print(f"🔄 批量推理：{len(indices_data)} 个指数 × {len(common_indices)} 天")
    
    all_metrics = []
    all_results = {}
    
    # 3. 逐天进行批量预测
    for idx in tqdm(common_indices):
        # 检查 lookback 是否充足
        if idx < config['lookback']:
            continue
        
        # 收集当前索引的所有指数数据
        batch_inputs = []
        batch_x_timestamps = []
        batch_y_timestamps = []
        batch_names = []
        batch_current_closes = []
        batch_current_dates = []
        batch_means = []
        batch_stds = []
        batch_dfs = []
        
        for name in indices_data.keys():
            df = indices_data[name]
            
            # 准备输入数据
            input_df = df.iloc[idx - config['lookback'] + 1 : idx + 1].copy()
            current_date = df.iloc[idx]["date"]
            current_close = df.iloc[idx]["close"]
            
            # 预处理（如果提供了预处理函数）
            if preprocess_fn is not None:
                x_norm, x_stamp, x_mean, x_std = preprocess_fn(input_df, config)
                norm_input_df = pd.DataFrame(x_norm, columns=config.get("feature_cols", ["open", "high", "low", "close", "volume"]))
                norm_input_df["date"] = input_df["date"].values
                batch_inputs.append(norm_input_df)
                batch_means.append(x_mean)
                batch_stds.append(x_std)
            else:
                # Base model 的情况：直接使用原始数据（predictor 内部会处理归一化）
                batch_inputs.append(input_df)
                batch_means.append(None)
                batch_stds.append(None)
            
            future_dates = pd.bdate_range(start=current_date + pd.Timedelta(days=1), periods=config['pred_len'])
            
            batch_x_timestamps.append(pd.Series(input_df["date"]))
            batch_y_timestamps.append(pd.Series(future_dates))
            batch_names.append(name)
            batch_current_closes.append(current_close)
            batch_current_dates.append(current_date)
            batch_dfs.append(df)
        
        # 4. 批量预测
        try:
            pred_outs = predictor.predict_batch(
                batch_inputs,
                batch_x_timestamps,
                batch_y_timestamps,
                pred_len=config['pred_len'],
                T=config['T'],
                top_p=config['top_p'],
                sample_count=config['sample_count'],
                verbose=False
            )
            
            # 5. 处理批量预测结果
            for i, name in enumerate(batch_names):
                pred_out = pred_outs[i]
                current_date = batch_current_dates[i]
                current_close = batch_current_closes[i]
                df = batch_dfs[i]
                
                row = {
                    "date": current_date,
                    "current_close": current_close,
                }
                
                for j in range(config['pred_len']):
                    # 获取预测值
                    if preprocess_fn is not None and denormalize_fn is not None:
                        # Finetuned 模型：需要反归一化
                        pred_z_score = pred_out.iloc[j]["close"]
                        pred_price = denormalize_fn(pred_z_score, batch_means[i], batch_stds[i], target_col_idx=3)
                    else:
                        # Base 模型：直接使用
                        pred_price = pred_out.iloc[j]["close"]
                    
                    row[f"pred_t+{j+1}"] = pred_price
                    
                    # Ground truth
                    if idx + 1 + j < len(df):
                        row[f"real_t+{j+1}"] = df.iloc[idx + 1 + j]["close"]
                    else:
                        row[f"real_t+{j+1}"] = np.nan
                
                # 存储预测结果
                if name not in all_results:
                    all_results[name] = []
                all_results[name].append(row)
                
        except Exception as e:
            print(f"⚠️ 批量预测在 idx={idx} 失败: {e}")
            pass
    
    # 6. 处理结果
    for name in all_results.keys():
        res_df = pd.DataFrame(all_results[name])
        # 保存
        res_df.to_csv(os.path.join(output_dir, f"predictions_{model_name}_{name}.csv"), index=False)
        
        # 计算指标
        idx_metrics = calculate_metrics(res_df)
        idx_metrics["Index"] = name
        all_metrics.append(idx_metrics)
        
        # 更新结果字典
        all_results[name] = res_df
    
    return all_metrics, all_results
