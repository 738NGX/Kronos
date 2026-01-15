import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

# ================= 配置 =================
RAW_SEC_ROOT = "/gemini/data-1/Tushare/sec"
RAW_INDEX_ROOT = "/gemini/data-1/Tushare/index"

OUTPUT_TRAIN_DIR = "/gemini/code/dataset/train"       
OUTPUT_INFER_DIR = "/gemini/code/dataset/inference"   

# 目标指数: 中证1000 (000852)
TARGET_INDEX_ID = "000852"
# =======================================

def get_column_map(data_coords):
    mapping = {}
    for col in data_coords:
        c_str = str(col).strip().upper()
        if 'OPEN' in c_str: mapping[col] = 'open'
        elif 'HIGH' in c_str: mapping[col] = 'high'
        elif 'LOW' in c_str: mapping[col] = 'low'
        elif 'CLOSE' in c_str: mapping[col] = 'close'
        elif 'VOL' in c_str: mapping[col] = 'volume'
        elif 'AMOUNT' in c_str: mapping[col] = 'amount'
    return mapping

def process_nc_file_final(file_path, output_dir, is_index=False):
    saved_count = 0
    try:
        # 1. 读取文件
        with xr.open_dataset(file_path, engine='netcdf4') as ds:
            # 2. 锁定维度 (基于你的探针结果)
            # Ticker: 股票代码, FDate: 日期, Data: 字段
            # 稍微做点兼容防止大小写不同
            dims = list(ds.dims)
            dim_ticker = next((d for d in dims if d in ['Ticker', 'ticker', 'Code', 'code']), None)
            dim_date = next((d for d in dims if d in ['FDate', 'fdate', 'Date', 'date', 'time']), None)
            dim_data = next((d for d in dims if d in ['Data', 'data']), None)
            
            if not (dim_ticker and dim_date and dim_data):
                # print(f"Skipped {os.path.basename(file_path)}: Dims mismatch {dims}")
                return 0

            # 3. 准备数据
            var_name = list(ds.data_vars)[0]
            da = ds[var_name]
            all_tickers = ds.coords[dim_ticker].values
            data_fields = ds.coords[dim_data].values
            col_mapping = get_column_map(data_fields)
            
            if not col_mapping: return 0

            # 4. 遍历 Ticker
            # 如果是微调个股，为了速度只取前 500 个 (足够复现)
            # 如果是找指数，则必须遍历所有
            loop_tickers = all_tickers if is_index else all_tickers[:500]

            for ticker in loop_tickers:
                # 代码已经是 str 格式 '000001.SZ'，直接用
                ticker_str = str(ticker).strip()
                
                # --- 过滤逻辑 ---
                if is_index:
                    # 只要包含 000852 就认为是中证1000
                    if TARGET_INDEX_ID not in ticker_str: continue
                    save_name = "000852.SH.csv"
                    print(f"   🎯 [命中] 找到中证1000: {ticker_str} 在 {os.path.basename(file_path)}")
                else:
                    # 个股: 只保留 .SZ / .SH
                    if not (ticker_str.endswith('SZ') or ticker_str.endswith('SH')):
                        continue
                    save_name = f"{ticker_str}.csv"

                # --- 提取 ---
                try:
                    # 切片
                    sub_da = da.sel({dim_ticker: ticker})
                    # 转 DataFrame
                    df = sub_da.to_dataframe().reset_index()
                    
                    # Pivot (处理重复索引)
                    # 关键修改：aggfunc='first' 避免 duplicate entries 报错
                    df_pivot = df.pivot_table(
                        index=dim_date, 
                        columns=dim_data, 
                        values=var_name, 
                        aggfunc='first'
                    ).reset_index()
                    
                    # 重命名列
                    df_pivot = df_pivot.rename(columns=col_mapping)
                    df_pivot = df_pivot.rename(columns={dim_date: 'timestamp'})
                    
                    # 检查必要列
                    if 'close' not in df_pivot.columns: continue

                    # --- 【关键修正】日期清洗 ---
                    # 之前死在这里。现在直接让 pandas 自动推断，不要指定 format
                    df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'], errors='coerce')
                    df_pivot = df_pivot.dropna(subset=['timestamp'])
                    
                    if len(df_pivot) < 5: continue

                    # 保存
                    save_path = os.path.join(output_dir, save_name)
                    hdr = not os.path.exists(save_path)
                    df_pivot.to_csv(save_path, mode='a', header=hdr, index=False)
                    saved_count += 1

                except Exception:
                    continue
                    
    except Exception as e:
        print(f"❌ 读取错误 {os.path.basename(file_path)}: {e}")
        
    return saved_count

def run():
    # 1. 准备目录
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_INFER_DIR, exist_ok=True)

    # 2. 准备文件列表
    print("🚀 正在扫描文件列表...")
    # 指数和个股文件都可能包含目标数据，全部纳入搜索范围
    all_nc_files = sorted(glob.glob(os.path.join(RAW_INDEX_ROOT, "**/*.nc"), recursive=True) + 
                          glob.glob(os.path.join(RAW_SEC_ROOT, "**/*.nc"), recursive=True))
    # 去重
    all_nc_files = sorted(list(set(all_nc_files)))
    
    if not all_nc_files:
        print("❌ 未找到任何 .nc 文件")
        return

    # 3. 提取中证1000 (优先任务)
    print(f"\n[Step 1] 寻找中证1000 ({len(all_nc_files)} 个文件)...")
    found_idx = False
    for f in tqdm(all_nc_files):
        cnt = process_nc_file_final(f, OUTPUT_INFER_DIR, is_index=True)
        if cnt > 0: found_idx = True
    
    if found_idx:
        print("   ✅ 中证1000 数据提取成功！")
    else:
        print("   ⚠️ 未找到中证1000数据 (稍后将自动生成模拟数据)")

    # 4. 提取个股 (用于微调)
    # 只处理 SEC 目录下的文件，且为了速度只处理前 20 个年份 (如 2005-2024)
    sec_only_files = sorted(glob.glob(os.path.join(RAW_SEC_ROOT, "**/*.nc"), recursive=True))
    
    print(f"\n[Step 2] 提取微调个股 (处理前 20 个年份文件)...")
    total_sec = 0
    
    # 进度条
    pbar = tqdm(sec_only_files[:20]) 
    for f in pbar:
        cnt = process_nc_file_final(f, OUTPUT_TRAIN_DIR, is_index=False)
        total_sec += cnt
        pbar.set_description(f"Extracted: {total_sec}")

    print(f"\n✅ 全部完成!")
    print(f"   -> 微调个股文件数: {len(glob.glob(os.path.join(OUTPUT_TRAIN_DIR, '*.csv')))}")
    
    # 最后简单去重
    print("   -> 正在执行最终去重...")
    for f in glob.glob(os.path.join(OUTPUT_TRAIN_DIR, "*.csv"))[:200]:
        try:
            df = pd.read_csv(f)
            df.drop_duplicates('timestamp', keep='last', inplace=True)
            df.to_csv(f, index=False)
        except: pass

if __name__ == "__main__":
    run()