import os
import sys
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
import torch

# Import shared utilities
from testutils.test_utils import (
    setup_environment, 
    process_index_results,
    aggregate_and_save_metrics,
    plot_all_results,
    parse_test_args
)
from testutils.common_config import INDICES
from testutils.data_utils import read_test_data, load_and_prepare_index_data

# Setup environment (fonts, paths, etc.)
setup_environment()

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    print("❌ Error: Could not import 'model'. Please run this script in the correct directory.")
    sys.exit(1)

# ================= Configuration =================
CONFIG = {
    "lookback": 400,
    "pred_len": 5,
    "T": 0.6,
    "top_p": 0.9,
    "sample_count": 10,
    "test_start": "2025-01-01",
    "test_end": "2025-09-30",
    "device": "cuda:0"
}

OUTPUT_DIR = "/gemini/code/outputs/base_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Main Logic =================

def run_reproduction(combine_plots=True):
    """
    运行基础模型测试
    
    Args:
        combine_plots: bool, True=拼成大图，False=独立输出每个指数图表
    """
    # 0. 一次性读取CSV文件
    all_data = read_test_data()
    
    # 1. Load Model (Load ONCE to save time)
    print(f"🚀 Loading Kronos Base Model on {CONFIG['device']}...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=CONFIG['device'], max_context=CONFIG['lookback'])
    
    all_metrics = []
    all_results = {}  # 存储所有指数的预测结果用于画图

    # 2. Iterate Indices
    for name, symbol in INDICES.items():
        print(f"\n{'='*10} Processing {name} {'='*10}")
        
        # 加载并预处理指数数据
        df, test_indices = load_and_prepare_index_data(all_data, name, symbol, CONFIG)
        if df is None:
            continue
        
        predictions = []
        
        # 3. Rolling Inference Loop
        # We need to predict every day in the test range
        print(f"🔄 Running Rolling Inference ({len(test_indices)} days)...")
        
        # Use tqdm for progress bar
        for idx in tqdm(test_indices):
            # Input window: [idx - lookback : idx]
            # Note: iloc excludes the end index for slicing, so we want up to idx (inclusive of current day's close)
            # Actually, standard logic: Input today's close -> Predict tomorrow
            if idx < CONFIG['lookback']:
                continue
                
            # Prepare Input Data
            # Data up to 'idx' (today). We want to predict idx+1, idx+2...
            input_df = df.iloc[idx - CONFIG['lookback'] + 1 : idx + 1]
            current_date = df.iloc[idx]["date"]
            current_close = df.iloc[idx]["close"]
            
            # Predict
            # We construct dummy y_timestamp just to satisfy the predictor signature
            # The predictor needs dates to form the output dataframe
            future_dates = pd.bdate_range(start=current_date + pd.Timedelta(days=1), periods=CONFIG['pred_len'])
            
            try:
                pred_out = predictor.predict(
                    df=input_df,
                    x_timestamp=pd.Series(input_df["date"]),
                    y_timestamp=pd.Series(future_dates),
                    pred_len=CONFIG['pred_len'],
                    T=CONFIG['T'],
                    top_p=CONFIG['top_p'],
                    sample_count=CONFIG['sample_count'] # Average of 10 samples
                )
                
                # pred_out contains mean/max/min. We usually take 'mean' or 'close' (if predictor averages automatically)
                # Assuming predictor returns a DF with 'close' which is the average prediction
                
                # Store T+1 to T+5 predictions
                row = {
                    "date": current_date,
                    "current_close": current_close,
                }
                
                # Map predictions to real values
                for i in range(CONFIG['pred_len']):
                    # Predicted Price
                    pred_price = pred_out.iloc[i]["close"]
                    row[f"pred_t+{i+1}"] = pred_price
                    
                    # Real Price (Look ahead in dataframe)
                    if idx + 1 + i < len(df):
                        real_price = df.iloc[idx + 1 + i]["close"]
                        row[f"real_t+{i+1}"] = real_price
                    else:
                        row[f"real_t+{i+1}"] = np.nan
                        
                predictions.append(row)
                
            except Exception as e:
                # print(f"Error on {current_date}: {e}")
                pass

        # 4. Save and Analyze Results for this Index
        res_df, idx_metrics = process_index_results(predictions, OUTPUT_DIR, "base", name)
        all_metrics.append(idx_metrics)
        all_results[name] = res_df

    # 6. Aggregate All Metrics into Summary Tables
    aggregate_and_save_metrics(all_metrics, OUTPUT_DIR, "base")
    
    # 7. 绘制预测曲线
    plot_all_results(all_results, OUTPUT_DIR, "base", CONFIG, combine_plots)

if __name__ == "__main__":
    args = parse_test_args('Test Kronos Base Model')
    run_reproduction(combine_plots=not args.separate_plots)