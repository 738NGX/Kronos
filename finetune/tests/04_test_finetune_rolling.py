"""
微调版 Kronos 滚动择时系统 (混合加速版)
- 策略: 双卡并行搜参 (提速) + 单卡串行推理 (稳定)
- 修复: 彻底解决 ThreadSafePredictor 的属性缺失问题
"""

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
from concurrent.futures import ThreadPoolExecutor

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
    
    # 双卡配置
    "gpu_ids": [0, 1], 
    
    "feature_cols": ["open", "high", "low", "close", "volume"],
    "time_feature_cols": ["minute", "hour", "weekday", "day", "month"],
    "clip_val": 3.0
}

PARAM_SEARCH_SPACE = {
    "T": [0.3, 0.6, 0.8, 1.0],
    "top_p": [0.2, 0.4, 0.6, 0.9],
    "lookback": [30, 60, 90]
}

OUTPUT_DIR = "/gemini/data-1/outputs/finetuned_rolling_hybrid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 优化器 (负责搜参) =================

class ParameterOptimizer:
    def __init__(self, predictor, config: Dict, device_id: int):
        self.predictor = predictor
        self.config = config
        self.device_id = device_id
    
    def grid_search(self, val_data: Dict[str, pd.DataFrame], param_space: Dict, val_start: pd.Timestamp, val_end: pd.Timestamp) -> Dict[str, Dict]:
        # 仅在 GPU 0 上打印进度，避免刷屏
        show_progress = (self.device_id == 0)
        
        # 强制上下文：确保搜参时的计算发生在正确的卡上
        with torch.cuda.device(self.device_id):
            if show_progress:
                print(f"\n🔍 [Parallel Search] 正在双卡优选参数... | 区间: {val_start.date()} ~ {val_end.date()}")

            min_data_len = min(len(df) for df in val_data.values())
            max_lookback_allowed = min_data_len - self.config["pred_len"] - 5
            filtered_lookbacks = [lb for lb in param_space["lookback"] if lb <= max_lookback_allowed]
            if not filtered_lookbacks: filtered_lookbacks = [30]

            final_space = {"T": param_space["T"], "top_p": param_space["top_p"], "lookback": filtered_lookbacks}
            param_names = list(final_space.keys())
            param_combinations = list(product(*final_space.values()))
            
            best_ic_map = {name: -2.0 for name in val_data.keys()}
            best_params_map = {name: None for name in val_data.keys()}

            iterator = tqdm(param_combinations, desc=f"   GPU {self.device_id} Scanning", leave=False, ncols=80) if show_progress else param_combinations
            
            for combo in iterator:
                params = dict(zip(param_names, combo))
                # 这一步只调用 self.predictor.predict，这是安全的
                scores_dict = self._evaluate_params_per_index(val_data, params, val_start, val_end)
                for name, score in scores_dict.items():
                    if score > best_ic_map[name]:
                        best_ic_map[name] = score
                        best_params_map[name] = params.copy()
            
            # 兜底
            default_params = {k: v[0] for k, v in final_space.items()}
            for name in val_data.keys():
                if best_params_map[name] is None: best_params_map[name] = default_params
                    
            return best_params_map

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
            
            sample_size = 4
            if len(valid_indices) > sample_size:
                step = len(valid_indices) // sample_size
                sampled_indices = valid_indices[::step]
            else:
                sampled_indices = valid_indices

            idx_preds, idx_actuals = [], []
            for idx in sampled_indices:
                if idx < lookback: continue
                real_input_len = lookback - 1
                input_df = df.iloc[idx - real_input_len + 1 : idx + 1]
                if len(input_df) != real_input_len: continue
                
                try:
                    with torch.no_grad():
                        # 直接调用原始 predict，无需 Wrapper，因为我们在外部 with torch.cuda.device 控制了上下文
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

# ================= 线程任务函数 (仅搜参) =================

def run_search_on_gpu(gpu_id: int, indices_subset: Dict, all_data: pd.DataFrame, 
                      val_start: str, val_end: str, predictor: KronosPredictor, optimizer: ParameterOptimizer):
    """
    只负责搜参，不负责推理
    """
    val_start_dt = pd.to_datetime(val_start)
    val_end_dt = pd.to_datetime(val_end)
    
    val_data_slice = {}
    max_lb = max(PARAM_SEARCH_SPACE['lookback'])
    history_buffer = pd.Timedelta(days=max_lb * 2)
    
    for name, symbol in indices_subset.items():
        df = all_data[all_data['代码'] == symbol].copy()
        mask = (df["时间"] >= (val_start_dt - history_buffer)) & (df["时间"] <= val_end_dt)
        slice_df = df[mask].rename(columns={
            "时间": "date", "开盘价(元)": "open", "最高价(元)": "high", 
            "最低价(元)": "low", "收盘价(元)": "close", "成交量(万股)": "volume"
        }).reset_index(drop=True)
        if not slice_df.empty:
            val_data_slice[name] = slice_df

    # 执行搜参
    return optimizer.grid_search(val_data_slice, PARAM_SEARCH_SPACE, val_start_dt, val_end_dt)

# ================= 主流程 =================

def run_hybrid_system():
    print("="*60)
    print("🚀 微调版 Kronos 滚动择时系统 (混合加速版)")
    print("   策略: 双卡并行搜参 + 单卡串行推理")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/5] 加载全量测试数据...")
    all_data = read_test_data()
    all_data["时间"] = pd.to_datetime(all_data["时间"])
    
    # 2. 初始化双路模型 (用于搜参)
    print("\n[2/5] 初始化双路模型 (cuda:0 & cuda:1)...")
    search_predictors = {}
    search_optimizers = {}
    
    tokenizer = KronosTokenizer.from_pretrained(CONFIG['tokenizer_path'])
    
    for gpu_id in CONFIG["gpu_ids"]:
        device = f"cuda:{gpu_id}"
        print(f"   ⚙️ Loading Search Model on {device}...")
        try:
            # 这里的模型用于搜参
            model = Kronos.from_pretrained(CONFIG['model_path']).to(device)
            pred = KronosPredictor(model, tokenizer, device=device, max_context=512)
            opt = ParameterOptimizer(pred, CONFIG, gpu_id)
            
            search_predictors[gpu_id] = pred
            search_optimizers[gpu_id] = opt
        except Exception as e:
            print(f"   ❌ Failed to load on {device}: {e}")
            return

    # 3. 任务分配
    all_indices_list = list(INDICES.items())
    mid_point = len(all_indices_list) // 2 + 1
    indices_groups = {
        0: dict(all_indices_list[:mid_point]), 
        1: dict(all_indices_list[mid_point:])
    }
    
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

    all_metrics_buffer = [] 
    global_pred_buffers = {name: [] for name in INDICES.keys()}
    global_params_history = [] 

    # 4. 滚动执行
    print("\n[3/5] 开始滚动流程...")
    
    # 线程池只用于搜参
    with ThreadPoolExecutor(max_workers=len(CONFIG["gpu_ids"])) as executor:
        
        for p_name, val_start, val_end, test_start, test_end in rolling_periods:
            print(f"\n📅 周期: {p_name}")
            
            # === STEP 1: 双卡并行搜参 ===
            # 这部分是最耗时的，我们并行跑
            futures = []
            for gpu_id in CONFIG["gpu_ids"]:
                future = executor.submit(
                    run_search_on_gpu,
                    gpu_id=gpu_id,
                    indices_subset=indices_groups[gpu_id],
                    all_data=all_data,
                    val_start=val_start,
                    val_end=val_end,
                    predictor=search_predictors[gpu_id],
                    optimizer=search_optimizers[gpu_id]
                )
                futures.append(future)
            
            # 收集参数结果
            current_period_best_params = {}
            for future in futures:
                try:
                    partial_params = future.result()
                    current_period_best_params.update(partial_params)
                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            global_params_history.append(current_period_best_params)
            
            # === STEP 2: 单卡串行推理 (使用 GPU 0 的模型) ===
            # 推理很快，没必要并行，避免复杂的 device 问题
            # 我们复用 search_predictors[0] 作为推理模型
            inference_predictor = search_predictors[0]
            
            # 打印本期优选结果
            print(f"   🏆 参数优选完成 -> 开始推理 ({test_start} ~ {test_end})")
            
            for name, symbol in INDICES.items():
                # 获取该指数的参数
                my_params = current_period_best_params.get(name)
                if not my_params:
                    # 兜底
                    my_params = {"T": CONFIG["T"], "top_p": CONFIG["top_p"], "lookback": CONFIG["lookback"]}
                
                # 构造 config (强制指定 device 为 cuda:0)
                idx_config = CONFIG.copy()
                idx_config.update(my_params)
                idx_config["test_start"] = test_start
                idx_config["test_end"] = test_end
                idx_config["device"] = "cuda:0" # 强制单卡推理
                
                single_index_dict = {name: symbol}
                
                # 调用原始 run_batch_inference (安全无毒)
                # 显式切换到 cuda:0 上下文，虽然 config 已经指定了，但双重保险
                with torch.cuda.device(0):
                    p_metrics, p_results = run_batch_inference(
                        predictor=inference_predictor,
                        all_data=all_data,
                        indices_dict=single_index_dict,
                        config=idx_config,
                        output_dir=OUTPUT_DIR,
                        model_name=f"temp_{p_name}_{name}", 
                        preprocess_fn=preprocess_window_finetuned,
                        denormalize_fn=denormalize,
                    )
                
                # 收集结果
                for m in p_metrics:
                    m["period"] = p_name
                    m["best_T"] = my_params["T"]
                    m["best_LB"] = my_params["lookback"]
                all_metrics_buffer.extend(p_metrics)
                
                if name in p_results:
                    global_pred_buffers[name].append(p_results[name])

    # 5. 汇总、拼接与保存
    print("\n[4/5] 汇总数据与绘图...")
    
    final_full_predictions = {}
    for name, df_list in global_pred_buffers.items():
        if df_list:
            full_df = pd.concat(df_list, axis=0).sort_values("date").reset_index(drop=True)
            final_full_predictions[name] = full_df
            save_path = os.path.join(OUTPUT_DIR, f"predictions_rolling_{name}.csv")
            full_df.to_csv(save_path, index=False)
            print(f"   💾 Saved: {save_path}")

    aggregate_and_save_metrics(all_metrics_buffer, OUTPUT_DIR, "rolling_all")
    plot_all_results(final_full_predictions, OUTPUT_DIR, "rolling_final", CONFIG, combine_plots=True)
    
    # 保存参数历史
    history_path = os.path.join(OUTPUT_DIR, "rolling_params_history.json")
    with open(history_path, 'w') as f:
        json.dump(global_params_history, f, indent=2, default=str)
    
    print(f"\n✅ 全部完成! 输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_hybrid_system()