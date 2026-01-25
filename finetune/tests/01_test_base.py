import os
import sys

# Import shared utilities
from testutils.test_utils import (
    setup_environment,
    run_batch_inference,
    aggregate_and_save_metrics,
    plot_all_results,
    parse_test_args
)
from testutils.common_config import INDICES
from testutils.data_utils import read_test_data

# Setup environment (fonts, paths, etc.)
setup_environment()

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

OUTPUT_DIR = "/gemini/code/outputs/base_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Main Logic =================

def run_reproduction(combine_plots=True):
    """
    运行基础模型测试
    
    Args:
        combine_plots: bool, True=拼成大图，False=独立输出每个指数图表
    """
    # 0. 一次性读取CSV文件
    all_data = read_test_data()
    
    # 1. Load Model (Load ONCE to save time)
    print(f"🚀 Loading Kronos Base Model on {CONFIG['device']}...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=CONFIG['device'], max_context=CONFIG['lookback'])
    
    # 2. 批量推理所有指数
    all_metrics, all_results = run_batch_inference(
        predictor=predictor,
        all_data=all_data,
        indices_dict=INDICES,
        config=CONFIG,
        output_dir=OUTPUT_DIR,
        model_name="base"
    )

    # 3. Aggregate All Metrics into Summary Tables
    aggregate_and_save_metrics(all_metrics, OUTPUT_DIR, "base")
    
    # 4. 绘制预测曲线
    plot_all_results(all_results, OUTPUT_DIR, "base", CONFIG, combine_plots)

if __name__ == "__main__":
    args = parse_test_args('Test Kronos Base Model')
    run_reproduction(combine_plots=not args.separate_plots)