import os
import pickle
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from tqdm import trange

from config import Config


class QlibDataPreprocessor:
    """
    A class to handle the loading, processing, and splitting of Qlib financial data.
    """

    def __init__(self):
        """Initializes the preprocessor with configuration and data fields."""
        self.config = Config()
        self.data_fields = ['open', 'close', 'high', 'low', 'volume', 'vwap']
        self.data = {}  # A dictionary to store processed data for each symbol.

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        print("Initializing Qlib...")
        qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)

    def load_qlib_data(self):
        """
        Loads raw data from Qlib, processes it symbol by symbol, and stores
        it in the `self.data` attribute.
        """
        print("Loading and processing data from Qlib...")
        data_fields_qlib = ['$' + f for f in self.data_fields]
        cal: np.ndarray = D.calendar()

        # Determine the actual start and end times to load, including buffer for lookback and predict windows.
        start_index = cal.searchsorted(pd.Timestamp(self.config.dataset_begin_time))
        end_index = cal.searchsorted(pd.Timestamp(self.config.dataset_end_time))

        # Check if start_index lookbackw_window will cause negative index
        adjusted_start_index = max(start_index - self.config.lookback_window, 0)
        real_start_time = pd.Timestamp(cal[adjusted_start_index])

        # Check if end_index exceeds the range of the array
        if end_index >= len(cal):
            end_index = len(cal) - 1
        elif cal[end_index] != pd.Timestamp(self.config.dataset_end_time):
            end_index -= 1

        # Check if end_index+predictw_window will exceed the range of the array
        adjusted_end_index = min(end_index + self.config.predict_window, len(cal) - 1)
        real_end_time = pd.Timestamp(cal[adjusted_end_index])

        # --- Hybrid Loading Logic ---
        # The split date for CSI1000 availability
        SPLIT_DATE = pd.Timestamp("2014-10-31")

        data_df = None
        
        # Trigger Condition: Instrument is csi1000 AND start time is before the official release
        if self.config.instrument == 'csi1000' and real_start_time < SPLIT_DATE:
            print("⚠️ Detected request for CSI1000 prior to 2014.")
            print("   -> Activating Hybrid Mode: splicing 'Proxy (Market-300-500)' and 'Real CSI1000'.")

            # Define splice points
            # Phase 1 ends at SPLIT_DATE
            # Phase 2 starts immediately after
            phase1_end = min(SPLIT_DATE, real_end_time)
            
            # --- Phase 1: Construct Proxy (Start -> 2014-10-31) ---
            print(f"   [Phase 1] Loading Proxy Data ({real_start_time.date()} to {phase1_end.date()})...")
            
            # Load Market, 300, and 500
            # Note: We only need indices for 300 and 500 to perform exclusion
            loader_all = QlibDataLoader(config=data_fields_qlib)
            df_all = loader_all.load('all', real_start_time, phase1_end)
            
            loader_filter = QlibDataLoader(config=['$close']) # Config doesn't matter, we just need the index
            df_300 = loader_filter.load('csi300', real_start_time, phase1_end)
            df_500 = loader_filter.load('csi500', real_start_time, phase1_end)
            
            # Perform Set Difference: All - (300 U 500)
            # Utilizing Pandas Index difference for speed
            exclude_idx = df_300.index.union(df_500.index)
            valid_idx = df_all.index.difference(exclude_idx)
            
            df_proxy = df_all.loc[valid_idx]
            print(f"   -> Phase 1 loaded. Raw: {len(df_all)}, Filtered (Proxy): {len(df_proxy)}")
            
            # --- Phase 2: Load Real CSI1000 (2014-11-01 -> End) ---
            df_real = pd.DataFrame()
            if real_end_time > SPLIT_DATE:
                phase2_start = SPLIT_DATE + pd.Timedelta(days=1)
                print(f"   [Phase 2] Loading Real Data ({phase2_start.date()} to {real_end_time.date()})...")
                df_real = QlibDataLoader(config=data_fields_qlib).load('csi1000', phase2_start, real_end_time)
                print(f"   -> Phase 2 loaded: {len(df_real)} records.")
            
            # Merge
            data_df = pd.concat([df_proxy, df_real])
            
            # Sort to ensure time order (crucial for time-series)
            data_df = data_df.sort_index()

        else:
            # Standard Loading Logic (Original)
            print(f"   -> Standard Mode: Loading {self.config.instrument} directly.")
            data_df = QlibDataLoader(config=data_fields_qlib).load(
                self.config.instrument, real_start_time, real_end_time
            )
        data_df = data_df.stack().unstack(level=1)  # Reshape for easier access.

        symbol_list = list(data_df.columns)
        for i in trange(len(symbol_list), desc="Processing Symbols"):
            symbol = symbol_list[i]
            symbol_df = data_df[symbol]

            # Pivot the table to have features as columns and datetime as index.
            symbol_df = symbol_df.reset_index().rename(columns={'level_1': 'field'})
            symbol_df = pd.pivot(symbol_df, index='datetime', columns='field', values=symbol)
            symbol_df = symbol_df.rename(columns={f'${field}': field for field in self.data_fields})

            # Calculate amount and select final features.
            symbol_df['vol'] = symbol_df['volume']
            symbol_df['amt'] = (symbol_df['open'] + symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 4 * symbol_df['vol']
            symbol_df = symbol_df[self.config.feature_list]

            # Filter out symbols with insufficient data.
            symbol_df = symbol_df.dropna()
            if len(symbol_df) < self.config.lookback_window + self.config.predict_window + 1:
                continue

            self.data[symbol] = symbol_df

    def prepare_dataset(self):
        """
        Splits the loaded data into train, validation, and test sets and saves them to disk.
        """
        print("Splitting data into train, validation, and test sets...")
        train_data, val_data, test_data = {}, {}, {}

        symbol_list = list(self.data.keys())
        for i in trange(len(symbol_list), desc="Preparing Datasets"):
            symbol = symbol_list[i]
            symbol_df = self.data[symbol]

            # Define time ranges from config.
            train_start, train_end = self.config.train_time_range
            val_start, val_end = self.config.val_time_range
            test_start, test_end = self.config.test_time_range

            # Create boolean masks for each dataset split.
            train_mask = (symbol_df.index >= train_start) & (symbol_df.index <= train_end)
            val_mask = (symbol_df.index >= val_start) & (symbol_df.index <= val_end)
            test_mask = (symbol_df.index >= test_start) & (symbol_df.index <= test_end)

            # Apply masks to create the final datasets.
            train_data[symbol] = symbol_df[train_mask]
            val_data[symbol] = symbol_df[val_mask]
            test_data[symbol] = symbol_df[test_mask]

        # Save the datasets using pickle.
        os.makedirs(self.config.dataset_path, exist_ok=True)
        with open(f"{self.config.dataset_path}/train_data.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{self.config.dataset_path}/val_data.pkl", 'wb') as f:
            pickle.dump(val_data, f)
        with open(f"{self.config.dataset_path}/test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)

        print("Datasets prepared and saved successfully.")


if __name__ == '__main__':
    # This block allows the script to be run directly to perform data preprocessing.
    preprocessor = QlibDataPreprocessor()
    preprocessor.initialize_qlib()
    preprocessor.load_qlib_data()
    preprocessor.prepare_dataset()

