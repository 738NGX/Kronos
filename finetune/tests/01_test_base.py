import os
import sys
from testutils.test_utils import (
    setup_environment,
    run_batch_inference,
    aggregate_and_save_metrics,
    plot_all_results,
    parse_test_args,
)
from testutils.common_config import INDICES, BASE_OUTPUT_DIR, BASE_CONFIG
from testutils.data_utils import read_test_data
from model import Kronos, KronosTokenizer, KronosPredictor

setup_environment()

CONFIG = BASE_CONFIG | { "device": "cuda:0" }

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "base_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_reproduction(combine_plots=True):
    """
    运行基础模型测试

    Args:
        combine_plots: bool, True=拼成大图，False=独立输出每个指数图表
    """
    all_data = read_test_data()

    print(f"🚀 Loading Kronos Base Model on {CONFIG['device']}...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(
        model, tokenizer, device=CONFIG["device"], max_context=CONFIG["lookback"]
    )

    all_metrics, all_results = run_batch_inference(
        predictor=predictor,
        all_data=all_data,
        indices_dict=INDICES,
        config=CONFIG,
        output_dir=OUTPUT_DIR,
        model_name="base",
    )
    aggregate_and_save_metrics(all_metrics, OUTPUT_DIR, "base")
    plot_all_results(all_results, OUTPUT_DIR, "base", CONFIG, combine_plots)


if __name__ == "__main__":
    args = parse_test_args("Test Kronos Base Model")
    run_reproduction(combine_plots=not args.separate_plots)
