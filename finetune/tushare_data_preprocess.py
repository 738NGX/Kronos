import os
import pickle
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm, trange
from config import Config

class QlibDataPreprocessor:
    """
    完全基于 Config 驱动的预处理器。
    不再硬编码任何路径，所有挂载点均从 self.config 获取。
    """

    def __init__(self, config_path: str):
        self.config = Config(config_path)
        # 内部映射 Tushare 的 Data 轴索引（OCHL + volume/amount + Adj + Turnover + MV）
        self.sec_map = {'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 7, 'amount': 8}
        self.bas_map = {'adj_factor': 1, 'turnover': 4, 'total_mv': 11}
        self.data = {}

    def initialize_qlib(self):
        """仅用于初始化输出环境"""
        os.makedirs(self.config.dataset_path, exist_ok=True)

    def _get_instrument_members(self):
        """
        按照源码逻辑，从 Config 指定的 qlib_data_path 下的 instruments 目录读取成员名单。
        """
        # Qlib 默认将 instrument 列表存放在 provider_uri/instruments/ 下
        member_path = os.path.join(self.config.qlib_data_path, 'instruments', f'{self.config.instrument}.txt')
        df = pd.read_csv(member_path, sep='\t', header=None, names=['ticker', 'start', 'end'])
        
        # 将 Qlib 格式 (SH600680) 转换为 Tushare 格式 (600680.SH)
        def to_ts_code(c):
            return f"{c[2:]}.{c[:2]}"
        df['ts_code'] = df['ticker'].apply(to_ts_code)
        df['start'] = pd.to_datetime(df['start'])
        df['end'] = pd.to_datetime(df['end'])
        return df

    def load_qlib_data(self):
        print(f"开始处理: {self.config.instrument}")
        
        # 严格引用 Config 时间
        start_t = pd.Timestamp(self.config.dataset_begin_time) - pd.Timedelta(days=self.config.lookback_window + 15)
        end_t = pd.Timestamp(self.config.dataset_end_time) + pd.Timedelta(days=self.config.predict_window + 15)

        # 1. 动态获取成员名单（路径完全由 Config 驱动）
        member_file = os.path.join(self.config.qlib_data_path, 'instruments', f'{self.config.instrument}.txt')
        members_df = pd.read_csv(member_file, sep='\t', header=None, names=['ticker', 'start', 'end'])
        
        # 统一代码格式转换
        members_df['ts_code'] = members_df['ticker'].apply(lambda x: f"{x[2:]}.{x[:2]}")
        members_df['start'] = pd.to_datetime(members_df['start'])
        members_df['end'] = pd.to_datetime(members_df['end'])
        
        target_tickers = members_df['ts_code'].unique()
        raw_storage = {t: [] for t in target_tickers}

        # 2. 扫描 Tushare 原始目录 (从 Config 获取 tushare_data_path)
        tushare_root = self.config.tushare_data_path
        years = range(start_t.year, end_t.year + 1)

        for year in tqdm(years, desc="Tushare 张量对齐提取"):
            sec_f = os.path.join(tushare_root, f"sec/checkpoint/{year}.nc")
            bas_f = os.path.join(tushare_root, f"basic/checkpoint/{year}.nc")
            if not (os.path.exists(sec_f) and os.path.exists(bas_f)): continue
            
            with xr.open_dataset(sec_f) as ds_sec_raw, xr.open_dataset(bas_f) as ds_bas_raw:
                # 获取坐标轴
                sec_tickers = ds_sec_raw.coords['Ticker'].values
                bas_tickers = ds_bas_raw.coords['Ticker'].values
                f_dates = pd.to_datetime(ds_sec_raw.coords['FDate'].values)
                
                # 获取核心数据变量名
                v_sec = ds_sec_raw['__xarray_dataarray_variable__']
                v_bas = ds_bas_raw['__xarray_dataarray_variable__']

                # --- 修复 IndexError 的核心逻辑：求交集并建立索引映射 ---
                # 找出在该年同时存在于名单、sec 和 basic 中的股票
                valid_tickers = np.intersect1d(target_tickers, np.intersect1d(sec_tickers, bas_tickers))
                
                for ticker in valid_tickers:
                    # 分别定位在两个张量中的物理索引，解决维度不一致问题
                    idx_s = np.where(sec_tickers == ticker)[0][0]
                    idx_b = np.where(bas_tickers == ticker)[0][0]
                    
                    # 提取复权因子并计算物理后复权
                    adj = v_bas.values[:, idx_b, self.bas_map['adj_factor']]
                    
                    node = pd.DataFrame({
                        'open': v_sec.values[:, idx_s, self.sec_map['open']] * adj,
                        'high': v_sec.values[:, idx_s, self.sec_map['high']] * adj,
                        'low': v_sec.values[:, idx_s, self.sec_map['low']] * adj,
                        'close': v_sec.values[:, idx_s, self.sec_map['close']] * adj,
                        'volume': v_sec.values[:, idx_s, self.sec_map['volume']],
                        'amount': v_sec.values[:, idx_s, self.sec_map['amount']],
                        'turnover': v_bas.values[:, idx_b, self.bas_map['turnover']],
                        'total_mv': v_bas.values[:, idx_b, self.bas_map['total_mv']]
                    }, index=f_dates)
                    
                    raw_storage[ticker].append(node[(node.index >= start_t) & (node.index <= end_t)])

        # 数据合并与动态成员切片
        for ticker, dfs in raw_storage.items():
            if not dfs: continue
            full_df = pd.concat(dfs).sort_index()
            full_df = full_df[~full_df.index.duplicated()]
            
            # 严格按照指数成员生效日期进行过滤（处理 2014 年前后的衔接）
            m_info = members_df[members_df['ts_code'] == ticker]
            mask = pd.Series(False, index=full_df.index)
            for _, row in m_info.iterrows():
                mask |= (full_df.index >= row['start']) & (full_df.index <= row['end'])
            
            # 最终字段筛选，严格对齐 Config.feature_list
            symbol_df = full_df[mask][self.config.feature_list].dropna()

            if len(symbol_df) >= self.config.lookback_window + self.config.predict_window + 1:
                # 强制将索引同步到列中，确保与 Qlib 脚本产出的格式 100% 一致
                symbol_df['datetime'] = symbol_df.index 
                self.data[ticker] = symbol_df

    def prepare_dataset(self):
        """严格依据 Config 定义的 train/val/test_time_range 划分并保存 pkl"""
        print("执行最终划分逻辑...")
        split_ranges = {
            'train': self.config.train_time_range,
            'val': self.config.val_time_range,
            'test': self.config.test_time_range
        }
        output = {k: {} for k in split_ranges.keys()}

        for symbol, df in self.data.items():
            for name, (s, e) in split_ranges.items():
                mask = (df.index >= s) & (df.index <= e)
                if mask.any():
                    output[name][symbol] = df[mask]

        for name, content in output.items():
            save_path = os.path.join(self.config.dataset_path, f"{name}_data.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(content, f)
            print(f"数据集已保存: {save_path}")

if __name__ == '__main__':
    # Usage: python tushare_data_preprocess.py --config path/to/config.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file (required)")
    args = parser.parse_args()

    preprocessor = QlibDataPreprocessor(config_path=args.config)
    preprocessor.initialize_qlib()
    preprocessor.load_qlib_data()
    preprocessor.prepare_dataset()