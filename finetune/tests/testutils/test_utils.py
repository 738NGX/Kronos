"""
通用测试工具函数
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def setup_environment():
    """配置测试环境（字体、路径等）"""
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
    
    # 2. 获取所有有效指数的测试索引（所有指数共同的日期）
    all_indices_set = [set(test_indices_dict[name]) for name in indices_data.keys()]
    common_indices = sorted(list(set.intersection(*all_indices_set)))
    
    if not common_indices:
        print("❌ No common test indices across all valid indices!")
        return [], {}
    
    # 按天批量（向量化）处理：每批包含多个天 × 多个指数
    batch_days = int(config.get('batch_days', 10))
    micro_batch_size = int(config.get('micro_batch_size', 64))
    chunks = [common_indices[i:i + batch_days] for i in range(0, len(common_indices), batch_days)]
    print(f"🔄 批量推理：{len(indices_data)} 个指数 × {len(common_indices)} 天，分为 {len(chunks)} 批，每批 {batch_days} 天")
    
    all_metrics = []
    all_results = {}
    
    for day_chunk in tqdm(chunks):
        # 组装一个大批次：包含 (指数数 × 天数) 条时间序列
        batch_inputs = []
        batch_x_timestamps = []
        batch_y_timestamps = []
        batch_names = []
        batch_current_closes = []
        batch_current_dates = []
        batch_means = []
        batch_stds = []
        batch_dfs = []
        batch_day_indices = []

        for name in indices_data.keys():
            df = indices_data[name]
            for idx in day_chunk:
                # 跳过不满足 lookback 的样本
                if idx < config['lookback']:
                    continue
                
                input_df = df.iloc[idx - config['lookback'] + 1 : idx + 1].copy()
                current_date = df.iloc[idx]["date"]
                current_close = df.iloc[idx]["close"]
                
                # 预处理（finetuned）或直接使用（base）
                if preprocess_fn is not None:
                    x_norm, x_stamp, x_mean, x_std = preprocess_fn(input_df, config)
                    norm_input_df = pd.DataFrame(x_norm, columns=config.get("feature_cols", ["open", "high", "low", "close", "volume"]))
                    norm_input_df["date"] = input_df["date"].values
                    batch_inputs.append(norm_input_df)
                    batch_means.append(x_mean)
                    batch_stds.append(x_std)
                else:
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
                batch_day_indices.append(idx)

        if not batch_inputs:
            continue

        # 统一批量预测（可切分为微批次以规避环境问题）
        preds_list = []
        total = len(batch_inputs)
        for start in range(0, total, micro_batch_size):
            end = min(start + micro_batch_size, total)
            sub_preds = predictor.predict_batch(
                batch_inputs[start:end],
                batch_x_timestamps[start:end],
                batch_y_timestamps[start:end],
                pred_len=config['pred_len'],
                T=config['T'],
                top_p=config.get('top_p', 0.9),
                sample_count=config.get('sample_count', 1),
                verbose=False
            )
            preds_list.extend(sub_preds)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # 将结果按指数分别收集
        for i in range(len(preds_list)):
            name = batch_names[i]
            df = batch_dfs[i]
            idx = batch_day_indices[i]
            current_date = batch_current_dates[i]
            current_close = batch_current_closes[i]
            pred_out = preds_list[i]
            
            row = {
                "date": current_date,
                "current_close": current_close,
            }
            for j in range(config['pred_len']):
                if preprocess_fn is not None and denormalize_fn is not None:
                    pred_z_score = pred_out.iloc[j]["close"]
                    pred_price = denormalize_fn(pred_z_score, batch_means[i], batch_stds[i], target_col_idx=3)
                else:
                    pred_price = pred_out.iloc[j]["close"]
                row[f"pred_t+{j+1}"] = pred_price
                
                if idx + 1 + j < len(df):
                    row[f"real_t+{j+1}"] = df.iloc[idx + 1 + j]["close"]
                else:
                    row[f"real_t+{j+1}"] = np.nan

            if name not in all_results:
                all_results[name] = []
            all_results[name].append(row)

    # 批次完成后：保存与评估
    from testutils.metrics_utils import calculate_metrics
    for name in all_results.keys():
        res_df = pd.DataFrame(all_results[name])
        res_df.to_csv(os.path.join(output_dir, f"predictions_{model_name}_{name}.csv"), index=False)
        idx_metrics = calculate_metrics(res_df)
        idx_metrics["Index"] = name
        all_metrics.append(idx_metrics)
        all_results[name] = res_df

    return all_metrics, all_results
