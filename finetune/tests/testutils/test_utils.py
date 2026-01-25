"""
通用测试工具函数
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def setup_environment():
    """配置测试环境（字体、路径等）"""
    # 设置中文字体
    plt.rcParams['font.family'] = 'Noto Serif CJK JP'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 确保模型路径可访问
    sys.path.append("/gemini/code/")


def save_prediction_results(predictions, output_dir, model_name, index_name):
    """
    保存单个指数的预测结果
    
    Args:
        predictions: list, 预测结果列表
        output_dir: str, 输出目录
        model_name: str, 模型名称 (base/finetuned)
        index_name: str, 指数名称
    
    Returns:
        pd.DataFrame: 预测结果DataFrame
    """
    res_df = pd.DataFrame(predictions)
    output_path = os.path.join(output_dir, f"predictions_{model_name}_{index_name}.csv")
    res_df.to_csv(output_path, index=False)
    return res_df


def aggregate_and_save_metrics(all_metrics, output_dir, model_name):
    """
    汇总所有指标并保存
    
    Args:
        all_metrics: list, 所有指数的指标列表
        output_dir: str, 输出目录
        model_name: str, 模型名称
    
    Returns:
        pd.DataFrame: 汇总后的指标DataFrame
    """
    from testutils.metrics_utils import save_and_print_metrics
    
    if all_metrics:
        final_df = pd.concat(all_metrics, ignore_index=True)
        save_and_print_metrics(final_df, output_dir, model_name=model_name)
        return final_df
    return None


def plot_all_results(all_results, output_dir, model_name, test_config, combine_subplots=True):
    """
    绘制所有指数的预测曲线
    
    Args:
        all_results: dict, 所有指数的预测结果
        output_dir: str, 输出目录
        model_name: str, 模型名称
        test_config: dict, 测试配置
        combine_subplots: bool, 是否组合为大图
    """
    from testutils.visualization_utils import plot_predictions
    
    if all_results:
        print("\n🎨 开始绘制预测曲线...")
        plot_predictions(all_results, output_dir, model_name=model_name, 
                        test_config=test_config, combine_subplots=combine_subplots)


def parse_test_args(description):
    """
    解析测试脚本的命令行参数
    
    Args:
        description: str, 脚本描述
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--separate-plots', action='store_true', 
                       help='输出独立图表而非组合大图')
    return parser.parse_args()


def process_index_results(predictions, output_dir, model_name, index_name):
    """
    处理单个指数的结果（保存+计算指标）
    
    Args:
        predictions: list, 预测结果列表
        output_dir: str, 输出目录
        model_name: str, 模型名称
        index_name: str, 指数名称
    
    Returns:
        tuple: (res_df, idx_metrics)
    """
    from testutils.metrics_utils import calculate_metrics
    
    # 保存结果
    res_df = save_prediction_results(predictions, output_dir, model_name, index_name)
    
    # 计算指标
    idx_metrics = calculate_metrics(res_df)
    idx_metrics["Index"] = index_name
    
    return res_df, idx_metrics
