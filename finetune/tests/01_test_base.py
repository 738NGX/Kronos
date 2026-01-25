import os
import sys
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
import torch

plt.rcParams['font.family'] = 'Noto Serif CJK JP'

# Ensure model path is accessible
sys.path.append("/gemini/code/") 
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

# Index Mapping (Akshare Symbols)
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

OUTPUT_DIR = "/gemeni/code/outputs/base_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Helper Functions =================

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

def calculate_metrics(results_df):
    """Calculate Spearman Corr and MAE for Price and Returns (T+1 to T+5)."""
    metrics = []
    
    # Iterate through horizons T+1 to T+5
    for step in range(1, 6):
        col_pred = f"pred_t+{step}"
        col_real = f"real_t+{step}"
        
        # Filter valid rows
        valid = results_df.dropna(subset=[col_pred, col_real])
        
        if len(valid) == 0:
            continue
            
        # 1. Price Metrics
        price_mae = np.mean(np.abs(valid[col_pred] - valid[col_real]))
        price_corr, _ = spearmanr(valid[col_pred], valid[col_real])
        
        # 2. Return Metrics
        # Real return: (Price_T+n / Price_T_current) - 1
        # Note: In the rolling log, 'current_close' is the close price on the inference day
        ret_real = (valid[col_real] / valid["current_close"]) - 1
        ret_pred = (valid[col_pred] / valid["current_close"]) - 1
        
        ret_mae = np.mean(np.abs(ret_real - ret_pred))
        ret_corr, _ = spearmanr(ret_real, ret_pred)
        
        metrics.append({
            "horizon": f"未来{step}日",
            "price_corr": price_corr,
            "price_mae": price_mae,
            "ret_corr": ret_corr,
            "ret_mae": ret_mae
        })
        
    return pd.DataFrame(metrics)

# ================= Main Logic =================

def run_reproduction():
    # 1. Load Model (Load ONCE to save time)
    print(f"🚀 Loading Kronos Model on {CONFIG['device']}...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=CONFIG['device'], max_context=CONFIG['lookback'])
    
    all_metrics = []

    # 2. Iterate Indices
    for name, symbol in INDICES.items():
        print(f"\n{'='*10} Processing {name} {'='*10}")
        df = get_index_data(symbol, name)
        if df is None: continue

        # Filter Date Range (Ensure enough lookback data exists)
        test_start_dt = pd.to_datetime(CONFIG['test_start'])
        test_end_dt = pd.to_datetime(CONFIG['test_end'])
        
        # Locate indices for the testing window
        mask = (df["date"] >= test_start_dt) & (df["date"] <= test_end_dt)
        test_indices = df[mask].index
        
        if len(test_indices) == 0:
            print(f"Skipping {name}: No data in test range.")
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
        res_df = pd.DataFrame(predictions)
        res_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_rolling_preds.csv"), index=False)
        
        # Calculate Metrics (Table 6-9 reproduction)
        idx_metrics = calculate_metrics(res_df)
        idx_metrics["Index"] = name
        all_metrics.append(idx_metrics)
        
        # 5. Plot Curves (Figure 10-15 reproduction)
        plot_rolling_curve(res_df, name)

    # 6. Aggregate All Metrics into Summary Tables
    if all_metrics:
        final_df = pd.concat(all_metrics)
        
        print("\n\n📊 ============ REPRODUCTION RESULTS ============")
        
        # Pivot for Price Correlation (Figure 6)
        fig6 = final_df.pivot(index="Index", columns="horizon", values="price_corr")
        print("\n[Figure 6] Price Correlation (Spearman):")
        print(fig6)
        fig6.to_csv(os.path.join(OUTPUT_DIR, "Fig6_Price_Correlation.csv"))
        
        # Pivot for Price MAE (Figure 7)
        fig7 = final_df.pivot(index="Index", columns="horizon", values="price_mae")
        print("\n[Figure 7] Price MAE:")
        print(fig7)
        fig7.to_csv(os.path.join(OUTPUT_DIR, "Fig7_Price_MAE.csv"))

        # Pivot for Return Correlation (Figure 8)
        fig8 = final_df.pivot(index="Index", columns="horizon", values="ret_corr")
        print("\n[Figure 8] Return Correlation (Spearman):")
        print(fig8)
        fig8.to_csv(os.path.join(OUTPUT_DIR, "Fig8_Return_Correlation.csv"))
        
        # Pivot for Return MAE (Figure 9)
        fig9 = final_df.pivot(index="Index", columns="horizon", values="ret_mae")
        print("\n[Figure 9] Return MAE:")
        print(fig9)
        fig9.to_csv(os.path.join(OUTPUT_DIR, "Fig9_Return_MAE.csv"))

def plot_rolling_curve(df, name):
    """Plot T+1 Prediction vs Ground Truth (Replicates Fig 10-15)."""
    plt.figure(figsize=(12, 6))
    
    # Plot Real Close
    # We use 'real_t+1' shifted back by 1 day to align visually with prediction date?
    # Actually, standard is: Plot(Time, Real_Close) vs Plot(Time, Pred_Close_for_Time)
    
    # Align dates: The row 'date' is the INFERENCE date. 'pred_t+1' is for date+1.
    # We want to plot the value on the day it actually happened.
    plot_dates = df["date"] + pd.Timedelta(days=1) # Shift to target date
    
    plt.plot(plot_dates, df["real_t+1"], label="Ground_truth", color="gray", alpha=0.7, linewidth=1.5)
    plt.plot(plot_dates, df["pred_t+1"], label="Prediction", color="#8B0000", linewidth=1.5) # Dark red matches CICC style
    
    plt.title(f"{name} Future 1-Day Price Prediction vs Ground Truth\n(2025.01 - 2025.09)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    path = os.path.join(OUTPUT_DIR, f"Fig_{name}_Prediction_Curve.png")
    plt.savefig(path)
    plt.close()
    print(f"📈 Saved chart to {path}")

if __name__ == "__main__":
    run_reproduction()