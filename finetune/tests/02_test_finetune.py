import os
from testutils.test_utils import (
    setup_environment,
    run_batch_inference,
    aggregate_and_save_metrics,
    plot_all_results,
    parse_test_args
)
from testutils.common_config import FINETUNE_CONFIG, INDICES, BASE_OUTPUT_DIR
from testutils.data_utils import read_test_data, preprocess_window_finetuned, denormalize
from model import Kronos, KronosTokenizer, KronosPredictor

setup_environment()

CONFIG = FINETUNE_CONFIG | { "device": "cuda:0" }

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
    print(f"🚀 Loading Finetuned Kronos from {CONFIG['model_path']}...")
    tokenizer = KronosTokenizer.from_pretrained(CONFIG['tokenizer_path'])
    model = Kronos.from_pretrained(CONFIG['model_path'])
    
    # 初始化预测器
    predictor = KronosPredictor(
        model, tokenizer, 
        device=CONFIG['device'], 
        max_context=CONFIG['lookback']
    )
    
    all_metrics = []
    all_results = {}  # 存储所有指数的预测结果用于画图

    # 2. 批量推理所有指数
    all_metrics, all_results = run_batch_inference(
        predictor=predictor,
        all_data=all_data,
        indices_dict=INDICES,
        config=CONFIG,
        output_dir=OUTPUT_DIR,
        model_name="finetuned",
        preprocess_fn=preprocess_window_finetuned,
        denormalize_fn=denormalize
    )

    # 3. 汇总保存
    aggregate_and_save_metrics(all_metrics, OUTPUT_DIR, "finetuned")
    
    # 4. 绘制预测曲线
    plot_all_results(all_results, OUTPUT_DIR, "finetuned", CONFIG, combine_plots)

if __name__ == "__main__":
    args = parse_test_args('Test Kronos Finetuned Model')
    run_inference(combine_plots=not args.separate_plots)