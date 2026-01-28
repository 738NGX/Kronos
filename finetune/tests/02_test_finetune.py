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
from testutils.common_config import FINETUNE_CONFIG, INDICES, BASE_OUTPUT_DIR
from testutils.data_utils import read_test_data

setup_environment()
from model import Kronos, KronosTokenizer, KronosPredictor

rank, local_rank, world_size = init_distributed_mode()

CONFIG = FINETUNE_CONFIG | { 
    "device": torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else "cpu"
}

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "finetuned_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= Main Logic =================

def run_inference(combine_plots=True):
    """
    运行微调模型测试
    
    Args:
        combine_plots: bool, True=拼成大图，False=独立输出每个指数图表
    """
    # 0. 一次性读取CSV文件
    all_data = read_test_data()
    
    # 1. 加载微调后的模型 (Safetensors)
    if rank == 0:
        print(f"🚀 Loading Finetuned Kronos from {CONFIG['model_path']}...")
    tokenizer = KronosTokenizer.from_pretrained(CONFIG['tokenizer_path'])
    model = Kronos.from_pretrained(CONFIG['model_path'])
    
    # 初始化预测器
    predictor = KronosPredictor(
        model, tokenizer, 
        device=CONFIG['device'], 
        max_context=CONFIG['lookback'],
        clip=CONFIG['clip_val']
    )
    
    all_metrics, all_results = run_distributed_inference(
        predictor=predictor,
        all_data=all_data,
        indices_dict=INDICES,
        config=CONFIG,
        output_dir=OUTPUT_DIR,
        model_name="finetuned",
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        aggregate_and_save_metrics(all_metrics, OUTPUT_DIR, "finetuned")
        plot_all_results(all_results, OUTPUT_DIR, "finetuned", CONFIG, combine_plots)

if __name__ == "__main__":
    args = parse_test_args('Test Kronos Finetuned Model')
    run_inference(combine_plots=not args.separate_plots)