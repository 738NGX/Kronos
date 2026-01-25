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

# 设置中文字体
plt.rcParams['font.family'] = 'Noto Serif CJK JP'
plt.rcParams['axes.unicode_minus'] = False

# 确保模型路径可访问
sys.path.append("/gemini/code/") 
from model import Kronos, KronosTokenizer, KronosPredictor

# Import visualization utilities
from utils.visualization_utils import plot_predictions

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

# 指数映射
INDICES = {
    "上证50": "000016.SH",
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
    "中证2000": "932000.CSI",
    "中证红利": "000922.CSI",
    "恒生指数": "HSI.HK",
    "恒生科技": "HSTECH.HK",
    "黄金ETF": "518880.SH",
}

OUTPUT_DIR = "/gemini/code/outputs/finetuned_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Data Processing =================

def get_index_data(symbol_code, name):
    print(f"📥 Loading {name} ({symbol_code}) from local iFind export...")
    try:
        full_df = pd.read_csv('/gemini/data-1/test_data.csv', thousands=',')

        df = full_df[full_df['代码'] == symbol_code].copy()
        
        df = df.rename(columns={
            "时间": "date",
            "开盘价(元)": "open",
            "最高价(元)": "high",
            "最低价(元)": "low",
            "收盘价(元)": "close",
            "成交量(万股)": "volume",
            "成交金额(万元)": "amount"
        })

        # Handle 'amount' (turnover)
        if "amount" not in df.columns:
            # Some akshare index interfaces lack amount, estimate it or leave simplified
            df["amount"] = df["close"] * df["volume"]
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # Type conversion
        cols = ["open", "high", "low", "close", "volume", "amount"]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
        return df
    except Exception as e:
        print(f"⚠️ Failed to fetch {name}: {e}")
        return None

def preprocess_window(df_window):
    """
    完全复刻训练时的预处理逻辑：
    1. 构造时间特征
    2. 执行实例级 Z-Score 归一化
    """
    # 1. 构造时间特征 (Time Embeddings)
    # 日线数据 minute/hour 通常为 0 或固定值，需与训练数据逻辑保持一致
    dates = df_window["date"].dt
    
    time_feats = pd.DataFrame({
        "minute": dates.minute,   # 日线通常为0
        "hour": dates.hour,       # 日线通常为0
        "weekday": dates.weekday,
        "day": dates.day,
        "month": dates.month
    })
    
    # 2. 提取基础特征
    x_raw = df_window[CONFIG["feature_cols"]].values.astype(np.float32)
    x_stamp = time_feats[CONFIG["time_feature_cols"]].values.astype(np.float32)
    
    # 3. 实例级归一化 (Instance-level Normalization)
    # 这是微调模型能工作的关键
    x_mean = np.mean(x_raw, axis=0)
    x_std = np.std(x_raw, axis=0)
    
    # 避免除零
    x_norm = (x_raw - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -CONFIG["clip_val"], CONFIG["clip_val"])
    
    return x_norm, x_stamp, x_mean, x_std

def denormalize(pred_norm, x_mean, x_std, target_col_idx=3):
    """
    反归一化：将模型输出的 Z-Score 还原为绝对价格
    target_col_idx=3 对应 'close' 列
    """
    # 还原公式: x = x_norm * std + mean
    # 注意：这里使用输入窗口的统计量进行还原
    return pred_norm * (x_std[target_col_idx] + 1e-5) + x_mean[target_col_idx]

# ================= Analysis Helper =================

def calculate_metrics(results_df):
    """计算 Spearman Corr 和 MAE"""
    metrics = []
    for step in range(1, CONFIG['pred_len'] + 1):
        col_pred = f"pred_t+{step}"
        col_real = f"real_t+{step}"
        
        valid = results_df.dropna(subset=[col_pred, col_real])
        if len(valid) == 0: continue
            
        # Price Metrics
        price_mae = np.mean(np.abs(valid[col_pred] - valid[col_real]))
        price_corr, _ = spearmanr(valid[col_pred], valid[col_real])
        
        # Return Metrics
        ret_real = (valid[col_real] / valid["current_close"]) - 1
        ret_pred = (valid[col_pred] / valid["current_close"]) - 1
        ret_mae = np.mean(np.abs(ret_real - ret_pred))
        ret_corr, _ = spearmanr(ret_real, ret_pred)
        
        metrics.append({
            "horizon": f"T+{step}",
            "price_corr": price_corr, "price_mae": price_mae,
            "ret_corr": ret_corr, "ret_mae": ret_mae
        })
    return pd.DataFrame(metrics)

# ================= Main Logic =================

def run_inference(combine_plots=True):
    """
    运行微调模型测试
    
    Args:
        combine_plots: bool, True=拼成大图，False=独立输出每个指数图表
    """
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
        df = get_index_data(symbol, name)
        
        # 确定测试区间索引
        test_start_dt = pd.to_datetime(CONFIG['test_start'])
        test_end_dt = pd.to_datetime(CONFIG['test_end'])
        mask = (df["date"] >= test_start_dt) & (df["date"] <= test_end_dt)
        test_indices = df[mask].index
        
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
            x_norm, x_stamp, x_mean, x_std = preprocess_window(input_df)
            
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

        # 保存结果
        res_df = pd.DataFrame(predictions)
        res_df.to_csv(os.path.join(OUTPUT_DIR, f"predictions_finetuned_{name}.csv"), index=False)
        
        # 计算指标
        idx_metrics = calculate_metrics(res_df)
        idx_metrics["Index"] = name
        all_metrics.append(idx_metrics)
        
        # 存储结果用于后续画图
        all_results[name] = res_df

    # 汇总保存
    if all_metrics:
        final_df = pd.concat(all_metrics, ignore_index=True)
        
        print("\n\n" + "="*60)
        print("📊 FINETUNED MODEL - EVALUATION RESULTS")
        print("="*60)
        
        # 保存完整指标表
        final_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_finetuned_all.csv"), index=False)
        print(f"\n✅ 完整指标已保存: metrics_finetuned_all.csv")
        
        # Pivot for Price Correlation
        price_corr = final_df.pivot(index="Index", columns="horizon", values="price_corr")
        print("\n[1] Price Correlation (Spearman):")
        print(price_corr.to_string())
        price_corr.to_csv(os.path.join(OUTPUT_DIR, "metrics_finetuned_price_correlation.csv"))
        
        # Pivot for Price MAE
        price_mae = final_df.pivot(index="Index", columns="horizon", values="price_mae")
        print("\n[2] Price MAE:")
        print(price_mae.to_string())
        price_mae.to_csv(os.path.join(OUTPUT_DIR, "metrics_finetuned_price_mae.csv"))

        # Pivot for Return Correlation
        ret_corr = final_df.pivot(index="Index", columns="horizon", values="ret_corr")
        print("\n[3] Return Correlation (Spearman):")
        print(ret_corr.to_string())
        ret_corr.to_csv(os.path.join(OUTPUT_DIR, "metrics_finetuned_return_correlation.csv"))
        
        # Pivot for Return MAE
        ret_mae = final_df.pivot(index="Index", columns="horizon", values="ret_mae")
        print("\n[4] Return MAE:")
        print(ret_mae.to_string())
        ret_mae.to_csv(os.path.join(OUTPUT_DIR, "metrics_finetuned_return_mae.csv"))
        
        print("\n" + "="*60)
        print(f"📁 所有结果文件已保存到: {OUTPUT_DIR}")
        print("="*60)
    
    # 7. 绘制预测曲线
    if all_results:
        print("\n🎨 开始绘制预测曲线...")
        plot_predictions(all_results, OUTPUT_DIR, model_name="finetuned", 
                        test_config=CONFIG, combine_subplots=combine_plots)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Kronos Finetuned Model')
    parser.add_argument('--separate-plots', action='store_true', 
                       help='输出独立图表而非组合大图')
    args = parser.parse_args()
    
    run_inference(combine_plots=not args.separate_plots)