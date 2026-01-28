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
    """
    从all_data中加载并预处理指定指数的数据
    
    Args:
        all_data: 包含所有指数数据的DataFrame
        name: 指数名称
        symbol: 指数代码
        config: 配置字典
    
    Returns:
        tuple: (df, test_indices) 或 (None, None) 如果数据无效
    """
    try:
        df = all_data[all_data['代码'] == symbol].copy()
        if df.empty:
            print(f"⚠️ Skipping {name}: No data found for code {symbol}")
            return None, None
        
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
        
        df['volume'] = df['volume'] * 10000
        df['amount'] = df['amount'] * 10000
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        cols = ["open", "high", "low", "close", "volume", "amount"]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        
        print(f"📥 Loaded {name} ({symbol}): {len(df)} records")
    except Exception as e:
        print(f"⚠️ Failed to process {name}: {e}")
        return None, None

    # 筛选测试日期范围（返回满足条件的行的位置索引）
    test_start_dt = pd.to_datetime(config['test_start'])
    test_end_dt = pd.to_datetime(config['test_end'])
    mask = (df["date"] >= test_start_dt) & (df["date"] <= test_end_dt)
    test_indices = np.where(mask)[0].tolist()  # 返回 True 位置的列表 [iloc_0, iloc_1, ...]
    
    if len(test_indices) == 0:
        print(f"Skipping {name}: No data in test range.")
        return None, None
    
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

