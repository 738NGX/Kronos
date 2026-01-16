"""
共享配置常量
"""

# 输出根目录
from unittest.mock import DEFAULT


BASE_OUTPUT_DIR = "/gemini/output"

# 指数映射 (Akshare Symbols)
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

BASE_CONFIG = {
    # 参数定义
    "lookback": 250,
    "pred_len": 5,
    "T": 0.6,
    "top_p": 0.9,
    "sample_count": 10,
    "clip_val": 5.0,  # tokenizer 和 KronosPredictor 的截断值
    # 测试范围
    "test_start": "2025-01-01",
    "test_end": "2025-12-31",
}
DEFAULT_MODEL = "/gemini/data-1/Kronos-base"
DEFAULT_TOKENIZER = "/gemini/data-1/Kronos-Tokenizer-base"

MODEL_REPO = "/gemini/data-1/outputs"
TOKENIZER_PATH = "finetune_tokenizer/checkpoints/best_model"
PREDICTOR_PATH = "finetune_predictor/checkpoints/best_model"

CSI300_TOKENIZER_PATH = f"{MODEL_REPO}/csi300_models_v2/{TOKENIZER_PATH}"
CSI500_TOKENIZER_PATH = f"{MODEL_REPO}/csi500_models_v2/{TOKENIZER_PATH}"
CSI1000_TOKENIZER_PATH = f"{MODEL_REPO}/csi1000_models/{TOKENIZER_PATH}"
CSI2000_TOKENIZER_PATH = f"{MODEL_REPO}/csi2000_models/{TOKENIZER_PATH}"

CSI300_MODEL_PATH = f"{MODEL_REPO}/csi300_models/{PREDICTOR_PATH}"
CSI500_MODEL_PATH = f"{MODEL_REPO}/csi500_models/{PREDICTOR_PATH}"
CSI1000_MODEL_PATH = f"{MODEL_REPO}/csi1000_models/{PREDICTOR_PATH}"
CSI2000_MODEL_PATH = f"{MODEL_REPO}/csi2000_models/{PREDICTOR_PATH}"

MODEL_CONFIG = {
    # 模型路径
    "model_path": {
        "上证50": CSI300_MODEL_PATH,
        "沪深300": CSI300_MODEL_PATH,
        "中证500": CSI500_MODEL_PATH,
        "中证1000": CSI1000_MODEL_PATH,
        "中证2000": CSI2000_MODEL_PATH,
        "中证红利": CSI500_MODEL_PATH,
        "恒生指数": "/gemini/data-1/finetune_csv/finetuned/hsi/basemodel/best_model",
        "恒生科技": "/gemini/data-1/finetune_csv/finetuned/hstech/basemodel/best_model",
        "黄金ETF": "/gemini/data-1/finetune_csv/finetuned/518880sh/basemodel/best_model",
        "default": DEFAULT_MODEL,
    },
    "tokenizer_path": {
        "上证50": CSI300_TOKENIZER_PATH,
        "沪深300": CSI300_TOKENIZER_PATH,
        "中证500": CSI500_TOKENIZER_PATH,
        "中证1000": CSI1000_TOKENIZER_PATH,
        "中证2000": CSI2000_TOKENIZER_PATH,
        "中证红利": CSI500_TOKENIZER_PATH,
        "恒生指数": DEFAULT_TOKENIZER,
        "恒生科技": DEFAULT_TOKENIZER,
        "黄金ETF": DEFAULT_TOKENIZER,
        "default": DEFAULT_TOKENIZER,
    },
    # 特征定义
    "feature_cols": ["open", "high", "low", "close", "volume", "amount"],
    "time_feature_cols": ["minute", "hour", "weekday", "day", "month"],
}

FINETUNE_CONFIG = BASE_CONFIG | MODEL_CONFIG
