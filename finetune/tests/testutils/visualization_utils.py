"""
Visualization utilities for Kronos model predictions
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_predictions(all_results, output_dir, model_name="base", test_config=None, combine_subplots=True):
    """
    绘制预测结果图表
    
    Args:
        all_results: dict, {index_name: results_df}
        output_dir: str, 输出目录路径
        model_name: str, 模型名称（用于标题和文件名）
        test_config: dict, 测试配置（包含test_start和test_end），可选
        combine_subplots: bool, True=拼成大图，False=独立输出每个指数
    """
    if not all_results:
        print("⚠️ 无可视化数据")
        return
    
    # 设置默认时间范围
    time_range = ""
    if test_config and "test_start" in test_config and "test_end" in test_config:
        time_range = f"\n({test_config['test_start']} to {test_config['test_end']})"
    
    if combine_subplots:
        # 拼图模式：所有指数作为子图
        n_indices = len(all_results)
        n_cols = 3
        n_rows = (n_indices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_indices > 1 else [axes]
        
        for idx, (name, df) in enumerate(all_results.items()):
            ax = axes[idx]
            # 将date转换为datetime类型并用作横坐标（预测值对应次日）
            plot_dates = pd.to_datetime(df["date"]) + pd.Timedelta(days=1)
            
            ax.plot(plot_dates, df["real_t+1"], label="Ground Truth", 
                   color="gray", alpha=0.7, linewidth=1.5)
            ax.plot(plot_dates, df["pred_t+1"], label="Prediction", 
                   color="#8B0000", linewidth=1.5)
            
            ax.set_title(f"{name}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel("Price", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 设置日期格式化器，确保标签与数据对齐
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for idx in range(len(all_results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f"T+1 Price Prediction - {model_name.upper()} Model{time_range}", 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"prediction_curves_{model_name}_combined.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 已保存组合图表: {save_path}")
        
    else:
        # 独立模式：每个指数单独保存
        for name, df in all_results.items():
            plt.figure(figsize=(12, 6))
            # 将date转换为datetime类型并用作横坐标（预测值对应次日）
            plot_dates = pd.to_datetime(df["date"]) + pd.Timedelta(days=1)
            
            plt.plot(plot_dates, df["real_t+1"], label="Ground Truth", 
                    color="gray", alpha=0.7, linewidth=1.5)
            plt.plot(plot_dates, df["pred_t+1"], label="Prediction", 
                    color="#8B0000", linewidth=1.5)
            
            plt.title(f"{name} - T+1 Price Prediction ({model_name.upper()} Model){time_range}", 
                     fontsize=14, fontweight='bold')
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Price", fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # 设置日期格式化器，确保标签与数据对齐
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"prediction_curve_{model_name}_{name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📈 已保存 {name} 图表")
        
        print(f"✅ 所有独立图表已保存到 {output_dir}")


def compare_models(base_results, finetuned_results, output_dir, test_config=None):
    """
    对比两个模型的预测结果
    
    Args:
        base_results: dict, 基础模型的结果 {index_name: results_df}
        finetuned_results: dict, 微调模型的结果 {index_name: results_df}
        output_dir: str, 输出目录路径
        test_config: dict, 测试配置（包含test_start和test_end），可选
    """
    if not base_results or not finetuned_results:
        print("⚠️ 需要同时提供两个模型的结果才能对比")
        return
    
    # 只对比两个模型都有的指数
    common_indices = set(base_results.keys()) & set(finetuned_results.keys())
    
    if not common_indices:
        print("⚠️ 两个模型没有共同的指数数据")
        return
    
    # 设置时间范围
    time_range = ""
    if test_config and "test_start" in test_config and "test_end" in test_config:
        time_range = f"\n({test_config['test_start']} to {test_config['test_end']})"
    
    n_indices = len(common_indices)
    n_cols = 3
    n_rows = (n_indices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_indices > 1 else [axes]
    
    for idx, name in enumerate(sorted(common_indices)):
        ax = axes[idx]
        
        base_df = base_results[name]
        fine_df = finetuned_results[name]
        
        plot_dates = base_df["date"] + pd.Timedelta(days=1)
        
        # 绘制真实值
        ax.plot(plot_dates, base_df["real_t+1"], label="Ground Truth", 
               color="gray", alpha=0.5, linewidth=1.5, linestyle='--')
        
        # 绘制基础模型预测
        ax.plot(plot_dates, base_df["pred_t+1"], label="Base Model", 
               color="#1f77b4", linewidth=1.2, alpha=0.8)
        
        # 绘制微调模型预测
        ax.plot(plot_dates, fine_df["pred_t+1"], label="Finetuned Model", 
               color="#d62728", linewidth=1.2, alpha=0.8)
        
        ax.set_title(f"{name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # 隐藏多余的子图
    for idx in range(len(common_indices), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"Model Comparison: Base vs Finetuned{time_range}", 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "model_comparison_combined.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 已保存模型对比图表: {save_path}")
