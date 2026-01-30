import os
import torch
from testutils.test_utils import (
    setup_environment,
    aggregate_and_save_metrics,
    plot_all_results,
    parse_test_args,
    init_distributed_mode,
    run_distributed_inference,
)
from testutils.common_config import INDICES, BASE_OUTPUT_DIR, BASE_CONFIG
from testutils.data_utils import read_test_data

setup_environment()
from model import Kronos, KronosTokenizer, KronosPredictor

rank, local_rank, world_size = init_distributed_mode()

CONFIG = BASE_CONFIG | { 
    "lookback": 400,
    "device": torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else "cpu"
}

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "base_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_reproduction(combine_plots=True):
    """
    运行基础模型测试

    Args:
        combine_plots: bool, True=拼成大图，False=独立输出每个指数图表
    """
    all_data = read_test_data()

    if rank == 0:
        print(f"🚀 Loading Kronos Base Model on {CONFIG['device']}...")
    
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(
        model, tokenizer, device=CONFIG["device"], max_context=512,
        clip=CONFIG['clip_val']
    )

    all_results = run_distributed_inference(
        predictor=predictor,
        all_data=all_data,
        indices_dict=INDICES,
        config=CONFIG,
        output_dir=OUTPUT_DIR,
        model_name="base",
        rank=rank,
        world_size=world_size,
    )
    
    if rank == 0:
        aggregate_and_save_metrics(all_results, OUTPUT_DIR, "base")
        plot_all_results(all_results, OUTPUT_DIR, "base", CONFIG, combine_plots)


if __name__ == "__main__":
    args = parse_test_args("Test Kronos Base Model")
    run_reproduction(combine_plots=not args.separate_plots)
