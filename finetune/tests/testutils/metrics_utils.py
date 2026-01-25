"""
Metrics calculation and reporting utilities
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def calculate_metrics(results_df, pred_len=5):
    """
    计算预测指标：Spearman相关系数和MAE（价格和收益率）
    
    Args:
        results_df: pd.DataFrame, 包含预测和真实值的结果数据框
        pred_len: int, 预测步长
    
    Returns:
        pd.DataFrame, 包含metrics的数据框
    """
    metrics = []
    
    for step in range(1, pred_len + 1):
        col_pred = f"pred_t+{step}"
        col_real = f"real_t+{step}"
        
        # 过滤有效行
        valid = results_df.dropna(subset=[col_pred, col_real])
        if len(valid) == 0:
            continue
        
        # 1. 价格指标
        price_mae = np.mean(np.abs(valid[col_pred] - valid[col_real]))
        price_corr, _ = spearmanr(valid[col_pred], valid[col_real])
        
        # 2. 收益率指标
        ret_real = (valid[col_real] / valid["current_close"]) - 1
        ret_pred = (valid[col_pred] / valid["current_close"]) - 1
        
        ret_mae = np.mean(np.abs(ret_real - ret_pred))
        ret_corr, _ = spearmanr(ret_real, ret_pred)
        
        metrics.append({
            "horizon": f"T+{step}",
            "price_corr": price_corr,
            "price_mae": price_mae,
            "ret_corr": ret_corr,
            "ret_mae": ret_mae
        })
    
    return pd.DataFrame(metrics)


def save_and_print_metrics(final_df, output_dir, model_name="base"):
    """
    保存和打印汇总指标
    
    Args:
        final_df: pd.DataFrame, 汇总指标数据框
        output_dir: str, 输出目录路径
        model_name: str, 模型名称（用于文件命名和打印）
    """
    print("\n\n" + "="*60)
    print(f"📊 {model_name.upper()} MODEL - EVALUATION RESULTS")
    print("="*60)
    
    # 保存完整指标表
    final_df.to_csv(os.path.join(output_dir, f"metrics_{model_name}_all.csv"), index=False)
    print(f"\n✅ 完整指标已保存: metrics_{model_name}_all.csv")
    
    # Pivot for Price Correlation
    price_corr = final_df.pivot(index="Index", columns="horizon", values="price_corr")
    print("\n[1] Price Correlation (Spearman):")
    print(price_corr.to_string())
    price_corr.to_csv(os.path.join(output_dir, f"metrics_{model_name}_price_correlation.csv"))
    
    # Pivot for Price MAE
    price_mae = final_df.pivot(index="Index", columns="horizon", values="price_mae")
    print("\n[2] Price MAE:")
    print(price_mae.to_string())
    price_mae.to_csv(os.path.join(output_dir, f"metrics_{model_name}_price_mae.csv"))
    
    # Pivot for Return Correlation
    ret_corr = final_df.pivot(index="Index", columns="horizon", values="ret_corr")
    print("\n[3] Return Correlation (Spearman):")
    print(ret_corr.to_string())
    ret_corr.to_csv(os.path.join(output_dir, f"metrics_{model_name}_return_correlation.csv"))
    
    # Pivot for Return MAE
    ret_mae = final_df.pivot(index="Index", columns="horizon", values="ret_mae")
    print("\n[4] Return MAE:")
    print(ret_mae.to_string())
    ret_mae.to_csv(os.path.join(output_dir, f"metrics_{model_name}_return_mae.csv"))
    
    print("\n" + "="*60)
    print(f"📁 所有结果文件已保存到: {output_dir}")
    print("="*60)
