"""
通用测试工具函数
"""
import os
import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import warnings


def init_distributed_mode():
    """初始化分布式推理/训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"🔥 [DDP] 进程启动: Global Rank {rank} | Local Rank {local_rank} | Total {world_size}")
        return rank, local_rank, world_size

    print("⚠️ [Single] 单卡模式运行")
    return 0, 0, 1

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
    
    # 忽略警告
    warnings.filterwarnings('ignore')

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


def run_distributed_inference(
    predictor,
    all_data,
    indices_dict,
    config,
    output_dir,
    model_name,
    rank: int = 0,
    world_size: int = 1,
    preprocess_fn=None,
    denormalize_fn=None,
):
    """
    通用多卡分布式推理

    Args:
        predictor: KronosPredictor 实例
        all_data: 所有指数的原始数据
        indices_dict: 指数名称->代码 字典
        config: 配置字典
        output_dir: 输出目录
        model_name: 模型名称 (base/finetuned)
        rank: 当前进程 rank
        world_size: 总进程数
        preprocess_fn: 可选预处理函数（finetuned 使用）
        denormalize_fn: 可选反归一化函数（finetuned 使用）

    Returns:
        tuple: (all_metrics, all_results) 仅 rank==0 时非空
    """
    from testutils.data_utils import load_and_prepare_index_data
    from testutils.metrics_utils import calculate_metrics

    if rank == 0:
        print("📥 加载所有指数数据...")

    indices_data = {}
    test_indices_dict = {}

    for name, symbol in indices_dict.items():
        df, test_indices = load_and_prepare_index_data(all_data, name, symbol, config)
        if df is not None:
            indices_data[name] = df
            test_indices_dict[name] = test_indices
        elif rank == 0:
            print(f"⚠️ Skipping {name}: 数据加载失败")

    if not indices_data:
        if rank == 0:
            print("❌ No valid index data found!")
        return [], {}

    # 基于日期而非行索引进行匹配
    test_dates_dict = {}
    for name in indices_data.keys():
        df = indices_data[name]
        # 只包含满足 lookback 要求的日期
        valid_dates = []
        for idx in test_indices_dict[name]:
            if idx >= config['lookback']:
                valid_dates.append(df.loc[idx, "date"])
        test_dates_dict[name] = set(valid_dates)
    
    all_dates_set = [test_dates_dict[name] for name in indices_data.keys()]
    common_dates = sorted(list(set.intersection(*all_dates_set)))

    if not common_dates:
        if rank == 0:
            print("❌ No common test dates across all valid indices!")
        return [], {}

    # 为每个指数建立日期到行索引的映射
    date_to_idx_map = {}
    for name in indices_data.keys():
        df = indices_data[name]
        date_to_idx_map[name] = {date: idx for idx, date in zip(df.index, df["date"])}

    my_dates = common_dates[rank::world_size]

    if rank == 0:
        print(f"🔄 分布式推理: {len(indices_data)} 个指数 × {len(common_dates)} 天")
        print(f"   ⚙️ 显卡数: {world_size} | 单卡任务: ~{len(my_dates)}")

    local_results = {name: [] for name in indices_data.keys()}

    import time
    start_time = time.time()
    log_interval = max(1, max(1, len(my_dates)) // 5)

    for i, current_date in enumerate(my_dates):
        batch_inputs = []
        batch_x_timestamps = []
        batch_y_timestamps = []
        batch_names = []
        batch_current_closes = []
        batch_current_dates = []
        batch_dfs = []
        batch_means = []
        batch_stds = []

        for name in indices_data.keys():
            df = indices_data[name]
            idx = date_to_idx_map[name].get(current_date)
            if idx is None or idx < config['lookback']:
                continue
            
            input_df = df.iloc[idx - config['lookback'] + 1 : idx + 1].copy()
            current_close = df.iloc[idx]["close"]

            future_dates = pd.bdate_range(start=current_date + pd.Timedelta(days=1), periods=config['pred_len'])

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

            batch_x_timestamps.append(pd.Series(input_df["date"]))
            batch_y_timestamps.append(pd.Series(future_dates))
            batch_names.append(name)
            batch_current_closes.append(current_close)
            batch_current_dates.append(current_date)
            batch_dfs.append(df)
        
        # 如果当前日期没有满足条件的指数，跳过
        if not batch_names:
            continue

        try:
            with torch.no_grad():
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

            for j, name in enumerate(batch_names):
                pred_out = pred_outs[j]
                current_date = batch_current_dates[j]
                current_close = batch_current_closes[j]
                df = batch_dfs[j]

                row = {
                    "date": current_date,
                    "current_close": current_close,
                }

                for k in range(config['pred_len']):
                    if preprocess_fn is not None and denormalize_fn is not None:
                        pred_z = pred_out.iloc[k]["close"]
                        pred_price = denormalize_fn(pred_z, batch_means[j], batch_stds[j], target_col_idx=3)
                    else:
                        pred_price = pred_out.iloc[k]["close"]

                    row[f"pred_t+{k+1}"] = pred_price

                    if idx + 1 + k < len(df):
                        row[f"real_t+{k+1}"] = df.iloc[idx + 1 + k]["close"]
                    else:
                        row[f"real_t+{k+1}"] = np.nan

                local_results[name].append(row)

        except Exception as e:
            if rank == 0:
                print(f"⚠️ 预测失败 date={current_date}: {e}")

        if (i + 1) % log_interval == 0 or (i + 1) == len(my_dates):
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * max(0, len(my_dates) - (i + 1))
            print(
                f"   🚀 [GPU-{rank}] 进度 {i+1:2d}/{max(1,len(my_dates))} ({((i+1)/max(1,len(my_dates)))*100:.0f}%) | "
                f"⏱️ {elapsed:.1f}s (剩 {remaining:.1f}s)"
            )

    if rank == 0:
        print("   ⏳ 等待其他 GPU 完成任务...")

    if world_size > 1:
        dist.barrier()

    if world_size > 1:
        all_results_list = [None for _ in range(world_size)]
        dist.all_gather_object(all_results_list, local_results)

        if rank == 0:
            merged_results = {name: [] for name in indices_data.keys()}
            for gpu_results in all_results_list:
                for name in indices_data.keys():
                    merged_results[name].extend(gpu_results[name])
            local_results = merged_results

    all_metrics = []
    final_results = {}

    if rank == 0:
        print("\n📊 汇总结果...")
        for name in indices_data.keys():
            if local_results[name]:
                res_df = pd.DataFrame(local_results[name])
                res_df = res_df.sort_values("date").reset_index(drop=True)

                save_path = os.path.join(output_dir, f"predictions_{model_name}_{name}.csv")
                res_df.to_csv(save_path, index=False)

                idx_metrics = calculate_metrics(res_df)
                idx_metrics["Index"] = name
                all_metrics.append(idx_metrics)

                final_results[name] = res_df
                print(f"   ✅ {name}: {len(res_df)} 条预测")

    return all_metrics, final_results
