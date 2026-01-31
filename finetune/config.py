import os
from typing import Any, Dict
import yaml

class Config:
    """
    Configuration class for the entire project.
    Supports loading overrides from a YAML config file.
    """
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        self._load_from_yaml(config_path)
        self._finalize()
    
    def _load_from_yaml(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            return
        if not isinstance(config, dict):
            raise ValueError("YAML config must be a mapping")
        self._apply_config_dict(config)

    def _apply_config_dict(self, config: Dict[str, Any]):
        section_map = {
            "data": [
                "qlib_data_path",
                "tushare_data_path",
                "instrument",
                "dataset_begin_time",
                "dataset_end_time",
                "lookback_window",
                "predict_window",
                "max_context",
                "feature_list",
                "time_feature_list",
            ],
            "dataset": [
                "train_time_range",
                "val_time_range",
                "test_time_range",
                "backtest_time_range",
                "dataset_path",
            ],
            "training": [
                "clip",
                "epochs",
                "log_interval",
                "batch_size",
                "n_train_iter",
                "n_val_iter",
                "tokenizer_learning_rate",
                "predictor_learning_rate",
                "accumulation_steps",
                "adam_beta1",
                "adam_beta2",
                "adam_weight_decay",
                "seed",
                "num_workers",
            ],
            "experiment": [
                "use_comet",
                "comet_config",
                "comet_tag",
                "comet_name",
            ],
            "paths": [
                "save_path",
                "tokenizer_save_folder_name",
                "predictor_save_folder_name",
                "backtest_save_folder_name",
                "backtest_result_path",
            ],
            "model": [
                "pretrained_repo",
                "pretrained_tokenizer_path",
                "pretrained_predictor_path",
                "finetuned_tokenizer_path",
                "finetuned_predictor_path",
            ],
            "backtest": [
                "backtest_n_symbol_hold",
                "backtest_n_symbol_drop",
                "backtest_hold_thresh",
                "inference_T",
                "inference_top_p",
                "inference_top_k",
                "inference_sample_count",
                "backtest_batch_size",
                "backtest_benchmark",
            ],
        }

        for section, keys in section_map.items():
            section_value = config.get(section)
            if isinstance(section_value, dict):
                for key in keys:
                    if key in section_value:
                        setattr(self, key, section_value[key])

        for key, value in config.items():
            if key in section_map:
                continue
            if hasattr(self, key):
                setattr(self, key, value)

    def _finalize(self):
        if self.n_train_iter is None:
            self.n_train_iter = 2000 * self.batch_size
        if self.n_val_iter is None:
            self.n_val_iter = 400 * self.batch_size

        format_ctx = {
            "save_path": self.save_path,
            "tokenizer_save_folder_name": self.tokenizer_save_folder_name,
            "predictor_save_folder_name": self.predictor_save_folder_name,
            "pretrained_repo": self.pretrained_repo,
        }
        for name in [
            "pretrained_tokenizer_path",
            "pretrained_predictor_path",
            "finetuned_tokenizer_path",
            "finetuned_predictor_path",
            "save_path",
            "backtest_result_path",
        ]:
            value = getattr(self, name, None)
            if isinstance(value, str) and "{" in value:
                try:
                    setattr(self, name, value.format(**format_ctx))
                except KeyError:
                    pass

        if not self.pretrained_tokenizer_path:
            self.pretrained_tokenizer_path = f"{self.pretrained_repo}/Kronos-Tokenizer-base"
        if not self.pretrained_predictor_path:
            self.pretrained_predictor_path = f"{self.pretrained_repo}/Kronos-base"
        if not self.finetuned_tokenizer_path:
            self.finetuned_tokenizer_path = (
                f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
            )
        if not self.finetuned_predictor_path:
            self.finetuned_predictor_path = (
                f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
            )
        if not self.backtest_benchmark:
            self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi500': "SH000905",
            'csi1000': "SH000852",
            'csi2000': "SH000852",
            'csi300': "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")
