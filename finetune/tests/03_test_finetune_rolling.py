import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from itertools import product
from typing import Dict, List, Optional
from tqdm import tqdm
from scipy.stats import spearmanr
import warnings
import contextlib

# 过滤警告
warnings.filterwarnings('ignore')

# 引入项目依赖
from testutils.test_utils import (
    setup_environment,
    run_batch_inference,
    aggregate_and_save_metrics,
    plot_all_results,
)
from testutils.common_config import INDICES
from testutils.data_utils import read_test_data, preprocess_window_finetuned, denormalize

# 初始化环境
setup_environment()

from model import Kronos, KronosTokenizer, KronosPredictor

import torch.distributed as dist

def init_distributed_mode():
    """初始化分布式环境，如果不是通过 torchrun 启动，则默认单卡运行"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # 核心：将当前进程绑定到指定的 GPU，防止张量乱跑
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        
        print(f"🔥 [DDP] 进程启动: Global Rank {rank} | Local Rank {local_rank} | Total {world_size}")
        return rank, local_rank, world_size
    else:
        print("⚠️ [Single] 单卡模式运行")
        return 0, 0, 1

# 在 setup_environment() 之后立即调用
rank, local_rank, world_size = init_distributed_mode()

# ================= 配置区域 =================

CONFIG = {
    "model_path": "/gemini/data-1/outputs/csi1000_models/finetune_predictor/checkpoints/best_model",
    "tokenizer_path": "/gemini/data-1/outputs/csi1000_models/finetune_tokenizer/checkpoints/best_model", 
    "lookback": 250,
    "pred_len": 5,
    "T": 0.6,
    "top_p": 0.9,
    "sample_count": 10,
    "test_start": "2025-01-01",
    "test_end": "2025-09-30",
    "device": torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else "cpu",
    "feature_cols": ["open", "high", "low", "close", "volume"],
    "time_feature_cols": ["minute", "hour", "weekday", "day", "month"],
    "clip_val": 3.0
}

PARAMS_CACHE_FILE = "/gemini/data-1/rolling_params_cache.json"

# 参数搜索空间
PARAM_SEARCH_SPACE = {
    "T": [0.3, 0.6, 0.8, 1.0],
    "top_p": [0.2, 0.4, 0.6, 0.9],
    "lookback": [30, 60, 90]
}

OUTPUT_DIR = "/gemini/data-1/outputs/finetuned_rolling_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 核心优化类 (保持不变) =================

class ParameterOptimizer:
    def __init__(self, predictor, config: Dict):
        self.predictor = predictor
        self.config = config
        self.rolling_params_history = []
        self.optimization_details = []
    
    def grid_search(self, val_data: Dict[str, pd.DataFrame], param_space: Dict, val_start: pd.Timestamp, val_end: pd.Timestamp) -> Dict[str, Dict]:
        # 仅主进程打印大的表头
        if rank == 0:
            print(f"\n🔍 [独立参数优化] 区间: {val_start.date()} ~ {val_end.date()}")

        min_data_len = min(len(df) for df in val_data.values())
        max_lookback_allowed = min_data_len - self.config["pred_len"] - 5
        filtered_lookbacks = [lb for lb in param_space["lookback"] if lb <= max_lookback_allowed]
        if not filtered_lookbacks: filtered_lookbacks = [30]

        final_space = {"T": param_space["T"], "top_p": param_space["top_p"], "lookback": filtered_lookbacks}
        param_names = list(final_space.keys())
        all_combinations = list(product(*final_space.values()))
        
        # === 🔥 并行切分 ===
        my_combinations = all_combinations[rank::world_size]
        total_tasks = len(my_combinations)
        
        if rank == 0:
            print(f"   ⚙️ 总组合: {len(all_combinations)} | 显卡数: {world_size} | 单卡任务: ~{total_tasks}")

        local_best_ic = {name: -2.0 for name in val_data.keys()}
        local_best_params = {name: None for name in val_data.keys()}

        log_interval = max(1, total_tasks // 5) 

        import time
        start_time = time.time()

        for i, combo in enumerate(my_combinations):
            params = dict(zip(param_names, combo))
            scores_dict = self._evaluate_params_per_index(val_data, params, val_start, val_end)
            
            for name, score in scores_dict.items():
                if score > local_best_ic[name]:
                    local_best_ic[name] = score
                    local_best_params[name] = params.copy()
            
            # --- 进度打印逻辑 ---
            # 只有在特定的步数，或者完成最后一步时才打印
            if (i + 1) % log_interval == 0 or (i + 1) == total_tasks:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (total_tasks - (i + 1))
                
                # 打印格式：[GPU-0] 进度 5/24 (20%) | 耗时 12s | 剩余 40s
                print(f"   🚀 [GPU-{rank}] 进度 {i+1:2d}/{total_tasks} ({((i+1)/total_tasks)*100:.0f}%) | "
                      f"⏱️ {elapsed:.1f}s (剩 {remaining:.1f}s)")

        # === 🔥 结果汇总 ===
        if rank == 0: print("   ⏳ 等待其他 GPU 完成任务...") # 提示用户进入等待阶段
        
        my_result = (local_best_params, local_best_ic)
        all_results_list = [None for _ in range(world_size)]
        
        if world_size > 1:
            try:
                dist.all_gather_object(all_results_list, my_result)
            except Exception as e:
                print(f"❌ [Rank {rank}] Gather Error: {e}")
                all_results_list = [my_result]
        else:
            all_results_list = [my_result]

        # === 决出全局最优 (仅 Rank 0) ===
        if rank == 0:
            final_best_params = {name: None for name in val_data.keys()}
            final_best_ic = {name: -2.0 for name in val_data.keys()}

            for gpu_params, gpu_ics in all_results_list:
                if gpu_params is None: continue 
                for name in val_data.keys():
                    if gpu_ics[name] > final_best_ic[name]:
                        final_best_ic[name] = gpu_ics[name]
                        final_best_params[name] = gpu_params[name]
            
            print("\n   🏆 [全局汇总] 各指数最优参数:")
            default_params = {k: v[0] for k, v in final_space.items()}
            for name in val_data.keys():
                if final_best_params[name] is None:
                    final_best_params[name] = default_params
                else:
                    p = final_best_params[name]
                    ic = final_best_ic[name]
                    print(f"     ✅ {name}: IC={ic:.4f} (T={p['T']}, LB={p['lookback']})")
            
            self.rolling_params_history.append(final_best_params)
            return final_best_params
        
        return {}

    def _evaluate_params_per_index(self, val_data, params, val_start, val_end):
        lookback = params["lookback"]
        pred_len = self.config["pred_len"]
        results = {}
        for name, df in val_data.items():
            mask = (df["date"] >= val_start) & (df["date"] <= val_end - pd.Timedelta(days=pred_len))
            valid_indices = df[mask].index.tolist()
            if not valid_indices:
                results[name] = -1.0
                continue
            
            total_days = len(valid_indices)
            
            if total_days <= 40:
                # 验证集很短时，为了 IC 准确，每一天都测
                step = 1
            else:
                # 验证集很长时，为了速度，限制最大样本数约为 20
                target_samples = 20
                step = max(1, total_days // target_samples)
            
            sampled_indices = valid_indices[::step]

            idx_preds, idx_actuals = [], []
            for idx in sampled_indices:
                if idx < lookback: continue
                real_input_len = lookback - 1
                input_df = df.iloc[idx - real_input_len + 1 : idx + 1]
                if len(input_df) != real_input_len: continue
                
                try:
                    with torch.no_grad():
                        pred_df = self.predictor.predict(
                            df=input_df, x_timestamp=input_df["date"],
                            y_timestamp=df.iloc[idx + 1 : idx + 1 + pred_len]["date"],
                            pred_len=pred_len, T=params["T"], top_p=params["top_p"],
                            sample_count=1, verbose=False
                        )
                    curr = input_df.iloc[-1]["close"]
                    pred = (pred_df.iloc[-1]["close"] - curr) / curr
                    act = (df.iloc[idx + pred_len]["close"] - curr) / curr
                    idx_preds.append(pred)
                    idx_actuals.append(act)
                except Exception: continue

            if len(idx_preds) < 2: results[name] = -1.0
            else:
                try:
                    corr, _ = spearmanr(idx_actuals, idx_preds)
                    results[name] = -1.0 if np.isnan(corr) else corr
                except: results[name] = -1.0
        return results

    def save_history(self, output_dir: str):
        path = os.path.join(output_dir, "rolling_params_history.json")
        with open(path, 'w') as f: json.dump(self.optimization_details, f, indent=2, default=str)
    
    def plot_evolution(self, output_dir: str):
        if not self.rolling_params_history: return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        history = self.rolling_params_history
        x = range(len(history))
        # 简单取第一个指数的参数做示意图
        first_idx = list(history[0].keys())[0]
        axes[0].plot(x, [p[first_idx]['T'] for p in history], 'o-', color='steelblue')
        axes[0].set_title(f'Temperature ({first_idx})')
        axes[1].plot(x, [p[first_idx]['top_p'] for p in history], 's-', color='orange')
        axes[1].set_title('Top-p')
        axes[2].plot(x, [p[first_idx]['lookback'] for p in history], '^-', color='green')
        axes[2].set_title('Lookback Window')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "params_evolution.png"))
        plt.close()

# ================= 主流程 =================

def run_rolling_system():
    print("="*60)
    print("🚀 微调版 Kronos 滚动择时系统 (最终拼接版 - 缓存优化 & 静默推理)")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/4] 加载全量测试数据...")
    all_data = read_test_data()
    all_data["时间"] = pd.to_datetime(all_data["时间"])
    
    # 2. 加载模型
    print("\n[2/4] 初始化模型...")
    try:
        tokenizer = KronosTokenizer.from_pretrained(CONFIG['tokenizer_path'])
        model = Kronos.from_pretrained(CONFIG['model_path'])
        predictor = KronosPredictor(
            model, tokenizer, device=CONFIG['device'], max_context=512
        )
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return

    optimizer = ParameterOptimizer(predictor, CONFIG)
    
    # 滚动周期配置
    rolling_periods = [
        ("2025.01", "2024-12-01", "2024-12-31", "2025-01-01", "2025-01-31"),
        ("2025.02", "2025-01-01", "2025-01-31", "2025-02-01", "2025-02-28"),
        ("2025.03", "2025-02-01", "2025-02-28", "2025-03-01", "2025-03-31"),
        ("2025.04", "2025-03-01", "2025-03-31", "2025-04-01", "2025-04-30"),
        ("2025.05", "2025-04-01", "2025-04-30", "2025-05-01", "2025-05-31"),
        ("2025.06", "2025-05-01", "2025-05-31", "2025-06-01", "2025-06-30"),
        ("2025.07", "2025-06-01", "2025-06-30", "2025-07-01", "2025-07-31"),
        ("2025.08", "2025-07-01", "2025-07-31", "2025-08-01", "2025-08-31"),
        ("2025.09", "2025-08-01", "2025-08-31", "2025-09-01", "2025-09-30"),
    ]

    # === 初始化全局容器 ===
    all_metrics_buffer = [] 
    global_pred_buffers = {name: [] for name in INDICES.keys()}

    # === 🔵 缓存加载逻辑 ===
    param_cache = {}
    if rank == 0:
        if os.path.exists(PARAMS_CACHE_FILE):
            try:
                with open(PARAMS_CACHE_FILE, 'r') as f:
                    param_cache = json.load(f)
                print(f"📦 [Cache] 已加载参数缓存文件: {PARAMS_CACHE_FILE}")
            except Exception as e:
                print(f"⚠️ [Cache] 缓存文件读取失败: {e}")

    # 3. 滚动执行
    print("\n[3/4] 开始滚动推理...")
    
    for p_name, val_start, val_end, test_start, test_end in rolling_periods:
        if rank == 0:
            print(f"\n📅 周期: {p_name}")
        
        # 准备验证数据
        val_start_dt = pd.to_datetime(val_start)
        val_end_dt = pd.to_datetime(val_end)
        
        # 提取验证集切片
        val_data_slice = {}
        max_lb = max(PARAM_SEARCH_SPACE['lookback'])
        history_buffer = pd.Timedelta(days=max_lb * 2) 
        
        for name, symbol in INDICES.items():
            df = all_data[all_data['代码'] == symbol].copy()
            mask = (df["时间"] >= (val_start_dt - history_buffer)) & (df["时间"] <= val_end_dt)
            slice_df = df[mask].rename(columns={
                "时间": "date", "开盘价(元)": "open", "最高价(元)": "high", 
                "最低价(元)": "low", "收盘价(元)": "close", "成交量(万股)": "volume"
            }).reset_index(drop=True)
            if not slice_df.empty:
                val_data_slice[name] = slice_df

        # --- A. 参数搜索 (带缓存检查) ---
        
        # 1. 决策：Rank 0 检查是否有缓存
        use_cache = False
        if rank == 0:
            if p_name in param_cache:
                use_cache = True
        
        # 2. 同步：Rank 0 将决策告诉 Rank 1 (防止 Rank 1 进入 GridSearch 等待)
        # 广播张量：1=Use Cache, 0=Search
        decision_tensor = torch.tensor([1 if use_cache else 0], dtype=torch.int, device=CONFIG['device'])
        if world_size > 1:
            dist.broadcast(decision_tensor, src=0)
        
        should_use_cache = (decision_tensor.item() == 1)

        # 3. 执行
        best_params_map = {}
        
        if should_use_cache:
            if rank == 0:
                print(f"   ⚡ [Cache] 命中缓存，跳过搜索")
                best_params_map = param_cache[p_name]
                # 简单打印一下参数
                sample_key = list(best_params_map.keys())[0]
                print(f"     👉 {sample_key}: {best_params_map[sample_key]}")
            # Rank 1 不需要 best_params_map，因为它会跳过推理
        else:
            # 没有缓存，正常执行并行搜索
            best_params_map = optimizer.grid_search(val_data_slice, PARAM_SEARCH_SPACE, val_start_dt, val_end_dt)
            
            # Rank 0 保存新参数到缓存
            if rank == 0:
                param_cache[p_name] = best_params_map
                try:
                    with open(PARAMS_CACHE_FILE, 'w') as f:
                        json.dump(param_cache, f, indent=2, default=str)
                    print(f"   💾 [Cache] 参数已更新并保存")
                except Exception as e:
                    print(f"   ⚠️ [Cache] 保存失败: {e}")

        # ⚠️ 同步屏障
        if world_size > 1:
            dist.barrier() 

        # === 只允许 Rank 0 进行后续的推理、绘图和保存 ===
        if rank != 0:
            continue 
        
        # --- B. 样本外测试 (静默版) ---
        print(f"   Run Inference: {test_start} -> {test_end}")
        
        for name, symbol in INDICES.items():
            my_params = best_params_map.get(name, best_params_map.get(list(best_params_map.keys())[0]))
            
            idx_config = CONFIG.copy()
            idx_config.update(my_params)
            idx_config["test_start"] = test_start
            idx_config["test_end"] = test_end
            
            single_index_dict = {name: symbol}
            
            # 🔵 使用 os.devnull 屏蔽 tqdm 的 stderr 输出
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull):
                    # 运行推理
                    p_metrics, p_results = run_batch_inference(
                        predictor=predictor,
                        all_data=all_data,
                        indices_dict=single_index_dict,
                        config=idx_config,
                        output_dir=OUTPUT_DIR,
                        model_name=f"temp_{p_name}_{name}", 
                        preprocess_fn=preprocess_window_finetuned,
                        denormalize_fn=denormalize,
                    )
            
            # 手动打印简单日志代替进度条
            print(f"     📝 [Inference] {name} 完成 ({len(p_results[name])} days)", flush=True)

            # 1. 收集 Metrics
            for m in p_metrics:
                m["period"] = p_name
                m["best_T"] = my_params["T"]
                m["best_LB"] = my_params["lookback"]
            all_metrics_buffer.extend(p_metrics)
            
            # 2. 收集 Predictions
            if name in p_results:
                global_pred_buffers[name].append(p_results[name])

    # 4. 汇总、拼接与保存 (Rank 0 only)
    if rank == 0:
        print("\n[4/4] 汇总数据与绘图...")
        
        # === 拼接所有月份的预测结果 ===
        final_full_predictions = {}
        
        for name, df_list in global_pred_buffers.items():
            if df_list:
                full_df = pd.concat(df_list, axis=0).sort_values("date").reset_index(drop=True)
                final_full_predictions[name] = full_df
                
                save_path = os.path.join(OUTPUT_DIR, f"predictions_rolling_{name}.csv")
                full_df.to_csv(save_path, index=False)
                print(f"   💾 已保存完整预测: {save_path} (Rows: {len(full_df)})")
            else:
                print(f"   ⚠️ 未收集到指数 {name} 的预测数据")

        # === 分离明细保存与汇总展示 ===
        if all_metrics_buffer:
            print(f"   📊 正在合并 {len(all_metrics_buffer)} 个分片 Metrics...")
            
            # 1. 纵向拼接所有分片
            df_metrics = pd.concat(all_metrics_buffer, ignore_index=True)
            
            # 2. 保存明细
            detail_save_path = os.path.join(OUTPUT_DIR, "rolling_metrics_detailed.csv")
            df_metrics.to_csv(detail_save_path, index=False)
            print(f"   💾 已保存分月明细指标: {detail_save_path}")

            # 3. 计算平均
            df_avg = df_metrics.groupby(['Index', 'horizon']).mean(numeric_only=True).reset_index()
            
            aggregate_and_save_metrics([df_avg], OUTPUT_DIR, "rolling_all_avg")
            
        else:
            print("   ⚠️ Metrics Buffer 为空，跳过汇总")
        
        # 保存搜参历史详情
        optimizer.save_history(OUTPUT_DIR)
        optimizer.plot_evolution(OUTPUT_DIR)
        
        print(f"\n✅ 全部完成! 输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_rolling_system()