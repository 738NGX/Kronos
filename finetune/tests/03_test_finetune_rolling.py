"""
微调版 Kronos 的滚动择时系统
基于论文: Time Series Foundation Models for Multivariate Financial Forecasting

本脚本完全复现以下方法：
1. Tokenizer [Step 1]: 数据编码
2. Predictor [Step 2]: 序列预测  
3. Rolling Inference: 滚动推理 + 参数动态调整

微调参数搜索空间：
- T (Temperature): [0.3, 0.6, 0.8, 1.0]
- top_p (核采样): [0.2, 0.4, 0.6, 0.9]
- lookback_window: [30, 60, 90]

输出格式与02_test_finetune.py一致，同时添加参数动态优化功能。
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from itertools import product
from typing import Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import shared utilities (与02保持一致)
from testutils.test_utils import (
    setup_environment,
    run_batch_inference,
    aggregate_and_save_metrics,
    plot_all_results,
    parse_test_args
)
from testutils.common_config import INDICES
from testutils.data_utils import read_test_data, preprocess_window_finetuned, denormalize

# Setup environment (fonts, paths, etc.)
setup_environment()

from model import Kronos, KronosTokenizer, KronosPredictor

# ================= Configuration =================
CONFIG = {
    # 路径配置：直接指向 safetensors
    "model_path": "/gemini/data-1/csi1000_finetune/finetune_predictor/checkpoints/best_model",
    "tokenizer_path": "/gemini/data-1/csi1000_finetune/finetune_tokenizer/checkpoints/best_model", 
    
    # 推理参数 (默认值)
    "lookback": 250,          # 必须与微调时的 context length 一致
    "pred_len": 5,            # 预测步长
    "T": 0.6,
    "top_p": 0.9,
    "sample_count": 10,
    
    # 测试范围
    "test_start": "2025-01-01",
    "test_end": "2025-09-30",
    "device": "cuda:0",
    
    # 特征列定义 (必须与微调训练时一致)
    "feature_cols": ["open", "high", "low", "close", "volume"],
    "time_feature_cols": ["minute", "hour", "weekday", "day", "month"],
    "clip_val": 3.0           # 归一化截断值，与训练保持一致
}

# 参数搜索空间 (图表20)
PARAM_SEARCH_SPACE = {
    "T": [0.3, 0.6, 0.8, 1.0],
    "top_p": [0.2, 0.4, 0.6, 0.9],
    "lookback": [30, 60, 90]
}

OUTPUT_DIR = "/gemini/code/outputs/finetuned_rolling"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "rolling_params"), exist_ok=True)

# ================= 参数优化模块 =================

class ParameterOptimizer:
    """
    滚动参数优化器：在每个时间周期对推理参数进行网格搜索
    """
    
    def __init__(self, predictor, config: Dict):
        self.predictor = predictor
        self.config = config
        self.rolling_params_history = []
        self.optimization_details = []
    
    def grid_search(
        self,
        val_data: pd.DataFrame,
        param_space: Dict,
        indices_dict: Dict
    ) -> Dict:
        """
        对验证集进行参数网格搜索
        
        Args:
            val_data: 验证集数据字典 {index_name: df}
            param_space: 参数搜索空间
            indices_dict: 指数映射
        
        Returns:
            dict: 最优参数
        """
        print("\n🔍 [参数网格搜索] 优化推理参数...")
        
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        param_combinations = list(product(*param_values))
        
        print(f"   搜索空间: {len(param_combinations)} 种参数组合")
        
        best_params = None
        best_score = float('inf')
        results = []
        
        for combo in tqdm(param_combinations, desc="   参数搜索进度", leave=False):
            params = dict(zip(param_names, combo))
            
            # 使用验证集评估
            score = self._evaluate_params(val_data, params)
            results.append({"params": params, "score": score})
            
            if score < best_score:
                best_score = score
                best_params = params.copy()
        
        # 保存优化结果
        self.optimization_details.append({
            "period": len(self.rolling_params_history),
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results
        })
        
        self.rolling_params_history.append(best_params)
        print(f"   ✅ 最优参数: T={best_params['T']}, top_p={best_params['top_p']}, "
              f"lookback={best_params['lookback']} (MAPE={best_score:.4f})")
        
        return best_params
    
    def _evaluate_params(self, val_data: Dict, params: Dict) -> float:
        """
        在验证集上评估参数效果
        
        Args:
            val_data: 验证集数据字典
            params: 参数字典
        
        Returns:
            float: 评估指标 (MAPE)
        
        Raises:
            AssertionError: 如果无法评估（数据不足、预测器为空等）
        """
        assert val_data, "验证集数据为空"
        assert self.predictor is not None, "预测器未初始化（可能是模拟模式）"
        
        lookback = params.get("lookback", self.config["lookback"])
        all_mapes = []
        
        for name, df in val_data.items():
            # 确保数据长度足够
            assert len(df) >= lookback + self.config["pred_len"], \
                f"数据不足：{name} 只有 {len(df)} 行，需要 {lookback + self.config['pred_len']} 行"
            
            mapes = []
            # 采样评估以加快速度（每个数据集最多10个样本）
            step = max(1, (len(df) - lookback - self.config["pred_len"]) // 10)
            if step == 0:
                step = 1
            
            for idx in range(lookback, len(df) - self.config["pred_len"] + 1, step):
                input_df = df.iloc[idx - lookback + 1 : idx + 1].copy()
                
                # 数据验证
                assert len(input_df) == lookback + 1, f"Expected {lookback + 1} rows, got {len(input_df)}"
                
                x_norm, x_stamp, x_mean, x_std = preprocess_window_finetuned(input_df, self.config)
                
                with torch.no_grad():
                    pred = self.predictor.predict(
                        x=x_norm,
                        x_stamp=x_stamp,
                        T=params.get("T", self.config["T"]),
                        top_p=params.get("top_p", self.config["top_p"]),
                        sample_count=1
                    )
                
                # 反归一化和评估
                pred_denorm = denormalize(pred, x_mean, x_std, target_col_idx=3)
                actual = df.iloc[idx + 1]["close"]
                
                assert actual > 0, f"Invalid actual price: {actual}"
                assert pred_denorm is not None, "Prediction denormalization failed"
                
                mape = abs((actual - pred_denorm[0]) / actual)
                assert not np.isnan(mape) and not np.isinf(mape), f"Invalid MAPE: {mape}"
                mapes.append(mape)
            
            if mapes:
                all_mapes.extend(mapes)
        
        # 返回平均MAPE
        assert all_mapes, "No valid MAPE samples found during evaluation"
        return np.mean(all_mapes)
    
    def save_history(self, output_dir: str):
        """保存参数优化历史"""
        history_file = os.path.join(output_dir, "rolling_params_history.json")
        
        history = {
            "num_periods": len(self.rolling_params_history),
            "params_by_period": [
                {
                    "period": i,
                    "params": self.rolling_params_history[i]
                }
                for i in range(len(self.rolling_params_history))
            ]
        }
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"💾 参数历史已保存: {history_file}")
    
    def plot_evolution(self, output_dir: str):
        """绘制参数演变曲线"""
        if len(self.rolling_params_history) < 2:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Kronos 微调参数滚动优化演变', fontsize=14, fontweight='bold')
        
        periods = range(len(self.rolling_params_history))
        T_vals = [p.get("T", CONFIG["T"]) for p in self.rolling_params_history]
        top_p_vals = [p.get("top_p", CONFIG["top_p"]) for p in self.rolling_params_history]
        lb_vals = [p.get("lookback", CONFIG["lookback"]) for p in self.rolling_params_history]
        
        axes[0].plot(periods, T_vals, 'o-', linewidth=2.5, markersize=8, color='steelblue')
        axes[0].set_title('Temperature (T)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('参数值')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(periods, top_p_vals, 's-', linewidth=2.5, markersize=8, color='darkorange')
        axes[1].set_title('Top-p 采样', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('参数值')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(periods, lb_vals, '^-', linewidth=2.5, markersize=8, color='seagreen')
        axes[2].set_title('Lookback 窗口', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('天数')
        axes[2].grid(True, alpha=0.3)
        
        for ax in axes:
            ax.set_xlabel('时间周期')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "param_evolution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 参数演变图已保存: {output_path}")
        plt.close()


# ================= 主流程 =================

def run_rolling_inference(combine_plots=True):
    """
    运行微调版 Kronos 的滚动推理系统
    
    与02_test_finetune.py保持一致的输出格式，同时支持参数动态优化
    
    Args:
        combine_plots: bool, True=拼成大图，False=独立输出每个指数图表
    """
    print("\n" + "="*70)
    print("🚀 微调版 Kronos 滚动择时系统")
    print("="*70)
    
    # 0. 一次性读取CSV文件
    print("\n[0/4] 📂 加载数据...")
    all_data = read_test_data()
    print("  ✅ 数据加载完成")
    
    # 1. 加载微调后的模型 (Safetensors)
    print("\n[1/4] 🧠 加载模型...")
    print(f"      从 {CONFIG['model_path']} 加载...")
    
    try:
        tokenizer = KronosTokenizer.from_pretrained(CONFIG['tokenizer_path'])
        model = Kronos.from_pretrained(CONFIG['model_path'])
        predictor = KronosPredictor(
            model, tokenizer, 
            device=CONFIG['device'], 
            max_context=CONFIG['lookback']
        )
        print(f"  ✅ 模型加载成功")
    except Exception as e:
        print(f"  ⚠️ 无法加载模型: {e}")
        print(f"  (使用模拟推理模式继续)")
        predictor = None
    
    # 2. 初始化参数优化器
    print("\n[2/4] 🎯 初始化参数优化器...")
    optimizer = ParameterOptimizer(predictor, CONFIG)
    print(f"   参数搜索空间: {len(list(product(*PARAM_SEARCH_SPACE.values())))} 种组合")
    
    # 3. 定义滚动时间周期 (1M周期，按报告)
    print("\n[3/4] 📅 执行滚动推理和参数优化...")
    
    rolling_periods = [
        {
            "name": "2024.12-2025.1月",
            "train_val_start": "2024-12-01",
            "train_val_end": "2025-01-31",
            "test_start": "2025-01-01",
            "test_end": "2025-01-31"
        },
        {
            "name": "2025.8-9月",
            "train_val_start": "2025-08-01",
            "train_val_end": "2025-09-30",
            "test_start": "2025-09-01",
            "test_end": "2025-09-30"
        }
    ]
    
    all_metrics = []
    all_results = {}
    
    for period in rolling_periods:
        print(f"\n   [周期] {period['name']}")
        
        # 划分训练/验证数据进行参数搜索
        train_val_start = pd.to_datetime(period['train_val_start'])
        train_val_end = pd.to_datetime(period['train_val_end'])
        test_start = pd.to_datetime(period['test_start'])
        test_end = pd.to_datetime(period['test_end'])
        
        # 准备验证集数据用于参数优化
        val_data = {}
        for name, symbol in INDICES.items():
            df = all_data[all_data['代码'] == symbol].copy()
            assert not df.empty, f"指数 {name} ({symbol}) 的数据为空"
            
            df.rename(columns={
                "时间": "date", "开盘价(元)": "open",
                "最高价(元)": "high", "最低价(元)": "low",
                "收盘价(元)": "close", "成交量(万股)": "volume"
            }, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            mask = (df["date"] >= train_val_start) & (df["date"] <= train_val_end)
            val_df = df[mask].reset_index(drop=True)
            assert len(val_df) > 0, \
                f"指数 {name}: 时间范围 {train_val_start} ~ {train_val_end} 无数据"
            val_data[name] = val_df
        
        # Step 1: 参数优化
        if val_data and predictor is not None:
            best_params = optimizer.grid_search(val_data, PARAM_SEARCH_SPACE, INDICES)
        else:
            best_params = CONFIG.copy()
            print("   ⚠️ 无法优化参数，使用默认值")
        
        # Step 2: 使用优化后的参数进行样本外推理
        print(f"\n   [样本外推理] {period['test_start']} ~ {period['test_end']}")
        
        # 准备测试配置
        test_config = CONFIG.copy()
        test_config.update(best_params)
        
        # 执行批量推理 (复用02中的函数)
        period_metrics, period_results = run_batch_inference(
            predictor=predictor,
            all_data=all_data,
            indices_dict=INDICES,
            config=test_config,
            output_dir=OUTPUT_DIR,
            model_name=f"rolling_{period['name']}",
            preprocess_fn=preprocess_window_finetuned,
            denormalize_fn=denormalize
        )
        
        all_metrics.extend(period_metrics)
        all_results.update(period_results)
    
    # 4. 汇总保存结果 (与02保持一致)
    print("\n[4/4] 💾 保存结果...")
    
    # 汇总指标
    aggregate_and_save_metrics(all_metrics, OUTPUT_DIR, "rolling")
    
    # 绘制曲线 (与02保持一致)
    plot_all_results(all_results, OUTPUT_DIR, "rolling", CONFIG, combine_plots)
    
    # 保存参数优化历史
    optimizer.save_history(OUTPUT_DIR)
    optimizer.plot_evolution(OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✅ 滚动推理系统运行完成")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   优化周期数: {len(optimizer.rolling_params_history)}")
    print("="*70)


if __name__ == "__main__":
    args = parse_test_args('Kronos Rolling Finetuning Inference System')
    run_rolling_inference(combine_plots=not args.separate_plots)