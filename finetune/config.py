import os

class Config:
    """
    Configuration class for the entire project.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # TODO: Update this path to your Qlib data directory.
        self.qlib_data_path = "/gemini/data-1/qlib/qlib_data/cn_data"
        self.tushare_data_path = "/gemini/data-2/Tushare/"
        self.instrument = 'csi1000'

        # Overall time range for data loading from Qlib.
        self.dataset_begin_time = "2010-01-01" 
        self.dataset_end_time = '2025-09-30'

        # Sliding window parameters for creating samples.
        self.lookback_window = 250  # Number of past time steps for input.
        self.predict_window = 5  # Number of future time steps for prediction.
        self.max_context = 512  # Maximum context length for the model.

        # Features to be used from the raw data.
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        # Time-based features to be generated.
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # Note: The validation/test set starts earlier than the training/validation set ends
        # to account for the `lookback_window`.
        self.train_time_range = ["2010-01-01", "2022-12-31"]
        self.val_time_range = ["2023-01-01", "2024-12-31"]
        self.test_time_range = ["2023-01-01", "2025-09-30"]
        self.backtest_time_range = ["2024-06-01", "2025-09-30"]

        # TODO: Directory to save the processed, pickled datasets.
        self.dataset_path = "/gemini/data-1/processed_datasets_csi1000"

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.

        self.epochs = 10
        self.log_interval = 50
        self.batch_size = 30

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        self.n_train_iter = 2000 * self.batch_size  # 60000
        self.n_val_iter = 400 * self.batch_size     # 12000

        # Learning rates for different model components.
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 1e-5

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = 1

        # AdamW optimizer parameters.
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # Miscellaneous
        self.seed = 100 # Global random seed for reproducibility.

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = False # Set to False if you don't want to use Comet ML

        # Base directory for saving model checkpoints and results.
        # Using a general 'outputs' directory is a common practice.
        self.save_path = "/gemini/data-1/outputs/csi1000_models"
        self.tokenizer_save_folder_name = 'finetune_tokenizer'
        self.predictor_save_folder_name = 'finetune_predictor'
        self.backtest_save_folder_name = 'finetune_backtest'

        # Path for backtesting results.
        self.backtest_result_path = "/gemini/data-1/outputs/csi1000_backtest_results"

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # These can be local paths or Hugging Face Hub model identifiers.
        self.pretrained_repo = "/gemini/data-1"
        self.pretrained_tokenizer_path = f"{self.pretrained_repo}/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = f"{self.pretrained_repo}/Kronos-base"

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50  # Number of symbols to hold in the portfolio.
        self.backtest_n_symbol_drop = 5  # Number of symbols to drop from the pool.
        self.backtest_hold_thresh = 5  # Minimum holding period for a stock.
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")
