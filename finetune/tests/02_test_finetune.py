import os
import sys
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
import torch
from safetensors.torch import load_file

# Import shared utilities
from testutils.test_utils import (
    setup_environment,
    process_index_results,
    aggregate_and_save_metrics,
    plot_all_results,
    parse_test_args
)
from testutils.common_config import INDICES
from testutils.data_utils import read_test_data, load_and_prepare_index_data, preprocess_window_finetuned, denormalize

# Setup environment (fonts, paths, etc.)
setup_environment()

from model import Kronos, KronosTokenizer, KronosPredictor

# ================= Configuration =================
CONFIG = {
    # 路径配置：直接指向 safetensors
    "model_path": "/gemini/data-1/csi1000_finetune/finetune_predictor/checkpoints/best_model", # 假设该目录下有 model.safetensors
    "tokenizer_path": "/gemini/data-1/csi1000_finetune/finetune_tokenizer/checkpoints/best_model", 
    
    # 推理参数
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

OUTPUT_DIR = "/gemini/code/outputs/finetuned_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Main Logic =================

def run_inference(combine_plots=True):
    """
    运行微调模型测试
    
    Args:
        combine_plots: bool, True=拼成大图，False=独立输出每个指数图表
    """
    # 0. 一次性读取CSV文件
    all_data = read_test_data()
    
    # 1. 加载微调后的模型 (Safetensors)
    print(f"🚀 Loading Finetuned Kronos from {CONFIG['model_path']}...")
    
    # 显式使用 safetensors 加载
    tokenizer = KronosTokenizer.from_pretrained(CONFIG['tokenizer_path'])
    # 注意：HuggingFace Transformer 库通常会自动识别 safetensors，
    # 但如果通过 use_safetensors=True 强制指定会更稳妥
    model = Kronos.from_pretrained(CONFIG['model_path'])
    
    # 初始化预测器
    predictor = KronosPredictor(
        model, tokenizer, 
        device=CONFIG['device'], 
        max_context=CONFIG['lookback']
    )
    
    all_metrics = []
    all_results = {}  # 存储所有指数的预测结果用于画图

    # 2. 遍历指数
    for name, symbol in INDICES.items():
        print(f"\n{'='*10} Processing {name} {'='*10}")
        
        # 加载并预处理指数数据
        df, test_indices = load_and_prepare_index_data(all_data, name, symbol, CONFIG)
        if df is None:
            continue
        
        predictions = []
        
        print(f"🔄 Rolling Inference on {len(test_indices)} days...")
        for idx in tqdm(test_indices):
            # 严格检查 lookback 长度
            # 如果 idx < lookback，数组切片会由 pandas 自动处理，但 context 长度不足
            # 此处不防御，假设数据充足，否则让其报错
            
            # 准备输入数据 (Raw Data)
            # 选取 [idx - lookback + 1 : idx + 1] 包含当前 idx 行
            input_df = df.iloc[idx - CONFIG['lookback'] + 1 : idx + 1].copy()
            current_date = df.iloc[idx]["date"]
            current_close = df.iloc[idx]["close"]
            
            # === 关键步骤：预处理与归一化 ===
            # 这里必须模拟 Code 2 Dataset 的 __getitem__ 逻辑
            x_norm, x_stamp, x_mean, x_std = preprocess_window_finetuned(input_df, CONFIG)
            
            # 构造预测用的时间戳 (用于 Positional Encoding)
            future_dates = pd.bdate_range(start=current_date + pd.Timedelta(days=1), periods=CONFIG['pred_len'])
            
            # 执行推理
            # 注意：传入的是 DataFrame 还是 Tensor 取决于 Predictor 实现
            # 假设 Predictor 内部能处理 DataFrame，我们传入归一化后的数据构造一个新的 DF
            # 为了适配 Predictor 接口，我们构造一个临时 DF
            norm_input_df = pd.DataFrame(x_norm, columns=CONFIG["feature_cols"])
            norm_input_df["date"] = input_df["date"].values # 保留日期列供 internal logic 使用
            
            pred_out = predictor.predict(
                df=norm_input_df, # 传入归一化后的数据
                x_timestamp=pd.Series(input_df["date"]),
                y_timestamp=pd.Series(future_dates),
                pred_len=CONFIG['pred_len'],
                T=CONFIG['T'],
                top_p=CONFIG['top_p'],
                sample_count=CONFIG['sample_count']
            )
            
            # === 关键步骤：结果反归一化 ===
            # pred_out 通常返回的是 'close' 列的预测值 (此时是 Z-Score)
            row = {
                "date": current_date,
                "current_close": current_close,
            }
            
            for i in range(CONFIG['pred_len']):
                # 获取归一化预测值
                pred_z_score = pred_out.iloc[i]["close"]
                
                # 还原为绝对价格
                pred_price_abs = denormalize(pred_z_score, x_mean, x_std, target_col_idx=3) # 3 is close
                row[f"pred_t+{i+1}"] = pred_price_abs
                
                # 获取 Ground Truth
                if idx + 1 + i < len(df):
                    row[f"real_t+{i+1}"] = df.iloc[idx + 1 + i]["close"]
                else:
                    row[f"real_t+{i+1}"] = np.nan
            
            predictions.append(row)

        # 保存结果并计算指标
        res_df, idx_metrics = process_index_results(predictions, OUTPUT_DIR, "finetuned", name)
        all_metrics.append(idx_metrics)
        all_results[name] = res_df

    # 汇总保存
    aggregate_and_save_metrics(all_metrics, OUTPUT_DIR, "finetuned")
    
    # 7. 绘制预测曲线
    plot_all_results(all_results, OUTPUT_DIR, "finetuned", CONFIG, combine_plots)

if __name__ == "__main__":
    args = parse_test_args('Test Kronos Finetuned Model')
    run_inference(combine_plots=not args.separate_plots)