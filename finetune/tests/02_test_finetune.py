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
    
    all_results = {}

    # 缓存（按路径复用）
    tokenizer_cache = {}
    model_cache = {}
    
    for index in INDICES:
        # 获取每个指数的模型路径
        if isinstance(CONFIG["model_path"], dict):
            model_path = CONFIG["model_path"].get(
                index,
                CONFIG["model_path"].get(
                    "default", list(CONFIG["model_path"].values())[0]
                ),
            )
        else:
            model_path = CONFIG["model_path"]

        # 获取每个指数对应的 tokenizer 路径
        if isinstance(CONFIG["tokenizer_path"], dict):
            tokenizer_path = CONFIG["tokenizer_path"].get(
                index,
                CONFIG["tokenizer_path"].get(
                    "default", list(CONFIG["tokenizer_path"].values())[0]
                ),
            )
        else:
            tokenizer_path = CONFIG["tokenizer_path"]

        if tokenizer_path not in tokenizer_cache:
            tokenizer_cache[tokenizer_path] = KronosTokenizer.from_pretrained(
                tokenizer_path
            )
        tokenizer = tokenizer_cache[tokenizer_path]

        if model_path not in model_cache:
            model_cache[model_path] = Kronos.from_pretrained(model_path)
        model = model_cache[model_path]
        
        # 初始化预测器
        predictor = KronosPredictor(
            model, tokenizer, 
            device=CONFIG['device'], 
            max_context=512,
            clip=CONFIG['clip_val']
        )
        
        results = run_distributed_inference(
            predictor=predictor,
            all_data=all_data,
            indices_dict={index: INDICES[index]},  # 只传递当前指数
            config=CONFIG,
            output_dir=OUTPUT_DIR,
            model_name=index,
            rank=rank,
            world_size=world_size,
        )
        all_results[index] = results  # 保存每个指数的结果

    if rank == 0:
        for index, results in all_results.items():
            aggregate_and_save_metrics(results, OUTPUT_DIR, index)
            plot_all_results(results, OUTPUT_DIR, index, CONFIG, combine_plots)

if __name__ == "__main__":
    args = parse_test_args('Test Kronos Finetuned Model')
    run_inference(combine_plots=not args.separate_plots)