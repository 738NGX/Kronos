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

    # 2. 批量推理所有指数
    all_metrics, all_results = run_batch_inference(
        predictor=predictor,
        all_data=all_data,
        indices_dict=INDICES,
        config=CONFIG,
        output_dir=OUTPUT_DIR,
        model_name="finetuned",
        preprocess_fn=preprocess_window_finetuned,
        denormalize_fn=denormalize
    )

    # 3. 汇总保存
    aggregate_and_save_metrics(all_metrics, OUTPUT_DIR, "finetuned")
    
    # 4. 绘制预测曲线
    plot_all_results(all_results, OUTPUT_DIR, "finetuned", CONFIG, combine_plots)

if __name__ == "__main__":
    args = parse_test_args('Test Kronos Finetuned Model')
    run_inference(combine_plots=not args.separate_plots)