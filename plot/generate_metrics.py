import pandas as pd
import numpy as np
import scipy.stats as stats
import os

# === 配置 ===
DATA_DIR = r"D:\Downloads\111" 
ASSET_NAMES = [
    "中证500", "中证1000", "中证2000", "恒生科技", 
    "沪深300", "上证50", "中证红利", "黄金ETF", "恒生指数"
]
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

def calc_strict_metrics(name):
    file_path = os.path.join(DATA_DIR, f"final_predictions_{name}_rank1.csv")
    if not os.path.exists(file_path): return None

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    mask = (df.index >= START_DATE) & (df.index <= END_DATE)
    df = df.loc[mask].copy()
    
    if len(df) == 0: return None

    # 计算收益率
    df['real_ret'] = df['current_close'].shift(-5) / df['current_close'] - 1
    df['pred_ret'] = df['pred_t+5'] / df['current_close'] - 1
    df.dropna(inplace=True)

    # 1. Rank IC (已定义)
    ic, _ = stats.spearmanr(df['pred_ret'], df['real_ret'])

    # 2. MAE (已定义) - 针对收益率计算，以便资产间可比
    # 如果是对绝对价格算MAE，恒生指数(20000点)和黄金(5元)没法比
    mae = np.mean(np.abs(df['pred_ret'] - df['real_ret']))

    # 3. 方向准确率 (新定义)
    same_dir = np.sign(df['pred_ret']) == np.sign(df['real_ret'])
    acc = same_dir.mean()

    return {
        "资产名称": name,
        "Rank IC": ic,     # 对应指标1
        "MAE": mae,        # 对应指标2
        "方向准确率": acc   # 对应指标3
    }

results = []
for asset in ASSET_NAMES:
    res = calc_strict_metrics(asset)
    if res: results.append(res)

res_df = pd.DataFrame(results)
# 格式化输出
print(res_df.to_string(formatters={
    'Rank IC': '{:.4f}'.format,
    'MAE': '{:.4f}'.format,
    '方向准确率': '{:.2%}'.format
}))