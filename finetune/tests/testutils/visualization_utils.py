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
        # 确保 axes 始终是一个一维可迭代的Axes对象列表
        # 当subplots返回ndarray时（即使是1D的(3,)形状），也需要flatten处理
        if hasattr(axes, 'flatten'):
            axes = axes.flatten()
        else:
            axes = [axes]
        
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
            
            # 明确设置横坐标范围，避免自动扩展
            if len(plot_dates) > 0:
                ax.set_xlim(plot_dates.min(), plot_dates.max())
                
                # 设置日期格式化器 - 显示具体日期
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
                
                # 根据起始日期生成刻度：每月的相应日期
                min_date = plot_dates.min()
                max_date = plot_dates.max()
                
                # 获取起始日期的日份
                start_day = min_date.day
                
                # 生成从起始日期开始，每月同一日期的刻度列表
                tick_dates = []
                current = min_date.replace(day=start_day)
                while current <= max_date:
                    if current >= min_date and current <= max_date:
                        tick_dates.append(current)
                    # 移到下一个月
                    try:
                        current = current.replace(month=current.month + 1)
                    except ValueError:
                        # 处理月份溢出（比如1月31日跳到2月）
                        if current.month == 12:
                            current = current.replace(year=current.year + 1, month=1)
                        else:
                            current = current.replace(month=current.month + 1)
                        # 如果该月没有这一天，用该月最后一天
                        if current.day < start_day:
                            import calendar
                            last_day = calendar.monthrange(current.year, current.month)[1]
                            current = current.replace(day=min(start_day, last_day))
                
                if tick_dates:
                    ax.set_xticks(tick_dates)
            
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
            
            # 明确设置横坐标范围，避免自动扩展
            ax = plt.gca()
            if len(plot_dates) > 0:
                ax.set_xlim(plot_dates.min(), plot_dates.max())
                
                # 设置日期格式化器 - 显示具体日期
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
                
                # 根据起始日期生成刻度：每月的相应日期
                min_date = plot_dates.min()
                max_date = plot_dates.max()
                
                # 获取起始日期的日份
                start_day = min_date.day
                
                # 生成从起始日期开始，每月同一日期的刻度列表
                tick_dates = []
                current = min_date.replace(day=start_day)
                while current <= max_date:
                    if current >= min_date and current <= max_date:
                        tick_dates.append(current)
                    # 移到下一个月
                    try:
                        current = current.replace(month=current.month + 1)
                    except ValueError:
                        # 处理月份溢出（比如1月31日跳到2月）
                        if current.month == 12:
                            current = current.replace(year=current.year + 1, month=1)
                        else:
                            current = current.replace(month=current.month + 1)
                        # 如果该月没有这一天，用该月最后一天
                        if current.day < start_day:
                            import calendar
                            last_day = calendar.monthrange(current.year, current.month)[1]
                            current = current.replace(day=min(start_day, last_day))
                
                if tick_dates:
                    ax.set_xticks(tick_dates)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"prediction_curve_{model_name}_{name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📈 已保存 {name} 图表")
        
        print(f"✅ 所有独立图表已保存到 {output_dir}")
