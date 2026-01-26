"""
微调版 Kronos 滚动择时系统 (优化版)
- 核心逻辑：基于 Spearman IC (收益率相关性) 进行参数优选
- 性能优化：验证集降频采样，大幅提升搜参速度
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

# 过滤掉 scipy 计算相关性时可能出现的除零警告
warnings.filterwarnings('ignore')

# 引入项目依赖 (请确保路径正确)
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
    # 路径配置
    "model_path": "/gemini/data-1/outputs/csi1000_models/finetune_predictor/checkpoints/best_model",
    "tokenizer_path": "/gemini/data-1/outputs/csi1000_models/finetune_tokenizer/checkpoints/best_model", 
    
    # 默认推理参数
    "lookback": 250,          # 默认窗口
    "pred_len": 5,            # 预测步长
    "T": 0.6,
    "top_p": 0.9,
    "sample_count": 10,       # 正式推理时的采样次数
    
    # 测试范围
    "test_start": "2025-01-01",
    "test_end": "2025-09-30",
    "device": "cuda:0",
    
    # 数据配置
    "feature_cols": ["open", "high", "low", "close", "volume"],
    "time_feature_cols": ["minute", "hour", "weekday", "day", "month"],
    "clip_val": 3.0
}

# 参数搜索空间 (与报告一致)
PARAM_SEARCH_SPACE = {
    "T": [0.3, 0.6, 0.8, 1.0],
    "top_p": [0.2, 0.4, 0.6, 0.9],
    "lookback": [30, 60, 90]
}

OUTPUT_DIR = "/gemini/code/outputs/finetuned_rolling_ic_optimized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 核心优化类 =================

class ParameterOptimizer:
    """
    滚动参数优化器 (IC Maximization Mode)
    """
    
    def __init__(self, predictor, config: Dict):
        self.predictor = predictor
        self.config = config
        self.rolling_params_history = []
        self.optimization_details = []
    
    def grid_search(
        self,
        val_data: Dict[str, pd.DataFrame],
        param_space: Dict,
        val_start: pd.Timestamp,
        val_end: pd.Timestamp
    ) -> Dict[str, Dict]:
        """
        对验证集进行参数网格搜索
        为每一个指数独立寻找 IC 最高的参数组合
        """
        print(f"\n🔍 [独立参数优化] 为9个指数分别寻找最佳参数 | 区间: {val_start.date()} ~ {val_end.date()}")

        # 1. 动态过滤 lookback (逻辑保持不变)
        min_data_len = min(len(df) for df in val_data.values())
        max_lookback_allowed = min_data_len - self.config["pred_len"] - 5
        filtered_lookbacks = [lb for lb in param_space["lookback"] if lb <= max_lookback_allowed]
        
        if not filtered_lookbacks:
            print(f"   ⚠️ 数据过短，使用默认 lookback=30")
            filtered_lookbacks = [30]

        final_space = {
            "T": param_space["T"],
            "top_p": param_space["top_p"],
            "lookback": filtered_lookbacks
        }
        
        param_names = list(final_space.keys())
        param_combinations = list(product(*final_space.values()))
        
        print(f"   ⚙️ 组合数: {len(param_combinations)} | 🎯 策略: 独立择优 (One-Pass Independent Selection)")

        # === 核心修改: 为每个指数初始化最佳记录 ===
        # 结构: {"上证50": -2.0, "沪深300": -2.0, ...}
        best_ic_map = {name: -2.0 for name in val_data.keys()}
        # 结构: {"上证50": {T:.., lookback:..}, ...}
        best_params_map = {name: None for name in val_data.keys()}

        # 2. 遍历所有参数组合
        for combo in tqdm(param_combinations, desc="   参数搜索", leave=False, ncols=100):
            params = dict(zip(param_names, combo))
            
            # 计算该参数在各个指数上的 IC
            # 返回字典: {"上证50": 0.15, "沪深300": 0.02, ...}
            scores_dict = self._evaluate_params_per_index(val_data, params, val_start, val_end)
            
            # 独立更新每个指数的最佳参数
            for name, score in scores_dict.items():
                if score > best_ic_map[name]:
                    best_ic_map[name] = score
                    best_params_map[name] = params.copy()
        
        # 3. 结果汇总与兜底
        print("\n   🏆 各指数最优参数结果:")
        default_params = {k: v[0] for k, v in final_space.items()}
        
        for name in val_data.keys():
            if best_params_map[name] is None:
                best_params_map[name] = default_params
                print(f"     ❌ {name}: 优化失败，使用默认")
            else:
                p = best_params_map[name]
                ic = best_ic_map[name]
                print(f"     ✅ {name}: IC={ic:.4f} | T={p['T']}, LB={p['lookback']}")
                
        # 记录优化历史 (简化存储，仅存 IC 均值或结构化数据)
        self.rolling_params_history.append(best_params_map)
        
        return best_params_map

    def _evaluate_params_per_index(
        self, 
        val_data: Dict[str, pd.DataFrame], 
        params: Dict, 
        val_start: pd.Timestamp, 
        val_end: pd.Timestamp
    ) -> Dict[str, float]:
        """
        计算给定参数在每个指数上的独立 IC
        """
        lookback = params["lookback"]
        pred_len = self.config["pred_len"]
        
        results = {}
        
        # 遍历所有指数 (不再使用代理指数，必须全跑)
        for name, df in val_data.items():
            mask = (df["date"] >= val_start) & (df["date"] <= val_end - pd.Timedelta(days=pred_len))
            valid_indices = df[mask].index.tolist()
            
            if not valid_indices:
                results[name] = -1.0
                continue
            
            # === 稀疏采样 (保持每指数采样4个点，保证速度) ===
            sample_size = 4
            if len(valid_indices) > sample_size:
                step = len(valid_indices) // sample_size
                sampled_indices = valid_indices[::step]
            else:
                sampled_indices = valid_indices

            idx_preds = []
            idx_actuals = []

            for idx in sampled_indices:
                if idx < lookback: continue
                
                # logic: lookback - 1
                real_input_len = lookback - 1
                input_df = df.iloc[idx - real_input_len + 1 : idx + 1]
                
                if len(input_df) != real_input_len: continue
                
                try:
                    with torch.no_grad():
                        pred_df = self.predictor.predict(
                            df=input_df,
                            x_timestamp=input_df["date"],
                            y_timestamp=df.iloc[idx + 1 : idx + 1 + pred_len]["date"],
                            pred_len=pred_len,
                            T=params["T"],
                            top_p=params["top_p"],
                            sample_count=1,
                            verbose=False
                        )
                    
                    # 收益率计算
                    curr = input_df.iloc[-1]["close"]
                    pred = (pred_df.iloc[-1]["close"] - curr) / curr
                    act = (df.iloc[idx + pred_len]["close"] - curr) / curr
                    
                    idx_preds.append(pred)
                    idx_actuals.append(act)
                    
                except Exception:
                    continue

            # 计算该指数的 IC
            if len(idx_preds) < 2: # 样本过少
                results[name] = -1.0
            else:
                try:
                    corr, _ = spearmanr(idx_actuals, idx_preds)
                    results[name] = -1.0 if np.isnan(corr) else corr
                except:
                    results[name] = -1.0
                    
        return results

    def save_history(self, output_dir: str):
        path = os.path.join(output_dir, "rolling_params_history.json")
        with open(path, 'w') as f:
            json.dump(self.optimization_details, f, indent=2, default=str)
        print(f"💾 参数历史保存至: {path}")

    def plot_evolution(self, output_dir: str):
        if not self.rolling_params_history: return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        history = self.rolling_params_history
        x = range(len(history))
        
        # Plot T
        axes[0].plot(x, [p['T'] for p in history], 'o-', color='steelblue')
        axes[0].set_title('Temperature (T)')
        axes[0].set_ylim(0, 1.1)
        
        # Plot Top_p
        axes[1].plot(x, [p['top_p'] for p in history], 's-', color='orange')
        axes[1].set_title('Top-p')
        axes[1].set_ylim(0, 1.1)

        # Plot Lookback
        axes[2].plot(x, [p['lookback'] for p in history], '^-', color='green')
        axes[2].set_title('Lookback Window')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "params_evolution.png"))
        plt.close()

# ================= 主流程 =================

def run_rolling_system():
    print("="*60)
    print("🚀 微调版 Kronos 滚动择时系统 (IC 优化版)")
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
            model, tokenizer, 
            device=CONFIG['device'], 
            max_context=512 # 确保足够大以容纳最大 lookback
        )
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return

    # 初始化优化器
    optimizer = ParameterOptimizer(predictor, CONFIG)
    
    # 3. 定义滚动周期 (每月滚动一次)
    # 逻辑：用上个月的数据做验证(搜参)，预测这个月
    rolling_periods = [
        # 格式: (周期名, 验证集开始, 验证集结束, 测试集开始, 测试集结束)
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

    all_metrics = []
    all_results = {}

    # 4. 滚动执行
    print("\n[3/4] 开始滚动推理...")
    
    for p_name, val_start, val_end, test_start, test_end in rolling_periods:
        print(f"\n📅 周期: {p_name}")
        
        # 准备验证数据
        val_start_dt = pd.to_datetime(val_start)
        val_end_dt = pd.to_datetime(val_end)
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)
        
        # 提取验证集切片 (需要包含足够的历史数据用于 lookback)
        val_data_slice = {}
        max_lb = max(PARAM_SEARCH_SPACE['lookback'])
        history_buffer = pd.Timedelta(days=max_lb * 2) 
        
        for name, symbol in INDICES.items():
            df = all_data[all_data['代码'] == symbol].copy()
            # 截取范围: (验证开始前N天) 到 (验证结束)
            mask = (df["时间"] >= (val_start_dt - history_buffer)) & (df["时间"] <= val_end_dt)
            slice_df = df[mask].rename(columns={
                "时间": "date", "开盘价(元)": "open", "最高价(元)": "high", 
                "最低价(元)": "low", "收盘价(元)": "close", "成交量(万股)": "volume"
            }).reset_index(drop=True)
            
            if not slice_df.empty:
                val_data_slice[name] = slice_df

        # --- A. 参数搜索 (搜参) ---
        # 此时返回的是 map: {"上证50": {params...}, "沪深300": {params...}}
        best_params_map = optimizer.grid_search(val_data_slice, PARAM_SEARCH_SPACE, val_start_dt, val_end_dt)
        
        # --- B. 样本外测试 (推理) ---
        print(f"   Running Inference: {test_start} -> {test_end}")
        
        current_metrics = []
        
        # 遍历所有指数，使用其专属参数进行预测
        for name, symbol in INDICES.items():
            # 获取该指数的专属参数 (如果没找到，用第一个作为兜底)
            my_params = best_params_map.get(name, best_params_map.get(list(best_params_map.keys())[0]))
            
            # 构造专属 config
            idx_config = CONFIG.copy()
            idx_config.update(my_params)
            idx_config["test_start"] = test_start
            idx_config["test_end"] = test_end
            
            # 构造单指数的 indices_dict
            single_index_dict = {name: symbol}
            
            # 运行单个指数的推理
            p_metrics, p_results = run_batch_inference(
                predictor=predictor,
                all_data=all_data,
                indices_dict=single_index_dict,
                config=idx_config,
                output_dir=OUTPUT_DIR,
                model_name=f"rolling_{p_name}_{name}", 
                preprocess_fn=preprocess_window_finetuned,
                denormalize_fn=denormalize
            )
            
            # 补充周期信息
            for m in p_metrics:
                m["period"] = p_name
                m["best_T"] = my_params["T"]
                m["best_LB"] = my_params["lookback"]
            
            current_metrics.extend(p_metrics)
            all_results.update(p_results)
            
        all_metrics.extend(current_metrics)

    # 5. 保存与绘图
    print("\n[4/4] 保存最终结果...")
    
    # 汇总保存
    aggregate_and_save_metrics(all_metrics, OUTPUT_DIR, "rolling_final")
    plot_all_results(all_results, OUTPUT_DIR, "rolling_final", CONFIG, combine_plots=True)
    
    optimizer.save_history(OUTPUT_DIR)
    optimizer.plot_evolution(OUTPUT_DIR)
    
    print(f"\n✅ 完成! 结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_rolling_system()