"""
Data loading and preprocessing utilities for test scripts
"""
import os
import pandas as pd
import numpy as np


def read_test_data(csv_path='/gemini/data-1/test_data.csv'):
    """
    从CSV文件加载指定指数的数据
    
    Args:
        symbol_code: str, 指数代码
        name: str, 指数名称（用于打印）
        csv_path: str, CSV文件路径
    
    Returns:
        pd.DataFrame or None, 处理后的数据框，若失败则返回None
    """
    print("📂 读取CSV文件...")
    for encoding in ['gbk', 'gb2312', 'gb18030', 'utf-8']:
        try:
            all_data = pd.read_csv(csv_path, thousands=',', encoding=encoding)
            print(f"✅ CSV读取成功 (编码: {encoding})")
            return all_data
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise ValueError("无法使用常见编码读取CSV文件")

def load_and_prepare_index_data(all_data, name, symbol, config):
    # 移除 try-catch，让错误直接抛出
    df = all_data[all_data['代码'] == symbol].copy()
    
    # 显式检查：如果数据为空，说明上游数据源有问题，直接报错而不是 print skipping
    if df.empty:
        raise ValueError(f"CRITICAL: No data found for code {symbol} ({name})")
    
    # 数据预处理
    df = df.rename(columns={
        "时间": "date",
        "开盘价(元)": "open",
        "最高价(元)": "high",
        "最低价(元)": "low",
        "收盘价(元)": "close",
        "成交量(万股)": "volume",
        "成交金额(万元)": "amount"
    })
    
    # 这里的缩放逻辑必须与主脚本严格一致
    df['volume'] = df['volume'] * 10000
    df['amount'] = df['amount'] * 10000
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    cols = ["open", "high", "low", "close", "volume", "amount"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") # errors='coerce' 可能会产生 NaN，需确认是否允许
        # 如果不允许 NaN，这里也应该去掉 coerce，直接崩溃
    
    # 筛选测试日期范围
    test_start_dt = pd.to_datetime(config['test_start'])
    test_end_dt = pd.to_datetime(config['test_end'])
    
    # 如果 config 中没有时间范围，这里会报错，符合预期
    mask = (df["date"] >= test_start_dt) & (df["date"] <= test_end_dt)
    test_indices = np.where(mask)[0].tolist()
    
    return df, test_indices

def preprocess_window_base(df_window, feature_cols=None):
    """
    基础模型的窗口预处理（可选，如果需要标准化）
    
    Args:
        df_window: pd.DataFrame, 输入数据窗口
        feature_cols: list, 特征列名称
    
    Returns:
        pd.DataFrame, 处理后的数据
    """
    return df_window

