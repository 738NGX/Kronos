import os
import json
import numpy as np
import pandas as pd
import torch
from itertools import product
from typing import Dict
from scipy.stats import spearmanr
from testutils.test_utils import (
    setup_environment,
    aggregate_and_save_metrics,
    plot_all_results,
    init_distributed_mode,
    run_distributed_inference,
)
from testutils.common_config import FINETUNE_CONFIG, INDICES, BASE_OUTPUT_DIR
from testutils.data_utils import read_test_data
import torch.distributed as dist

setup_environment()
from model import Kronos, KronosTokenizer, KronosPredictor

rank, local_rank, world_size = init_distributed_mode()

CONFIG = FINETUNE_CONFIG | { "device": torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else "cpu" }

PARAMS_CACHE_FILE = "/gemini/data-1/rolling_params_cache.json"
PARAM_SEARCH_SPACE = {
    "T": [0.3, 0.6, 0.8, 1.0],
    "top_p": [0.2, 0.4, 0.6, 0.9],
    "lookback": [30, 60, 90]
}

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "finetuned_rolling_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ParameterOptimizer:
    def __init__(self, predictor, config: Dict):
        self.predictor = predictor
        self.config = config
    
    def grid_search(self, val_data: Dict[str, pd.DataFrame], param_space: Dict, val_start: pd.Timestamp, val_end: pd.Timestamp) -> Dict[str, Dict]:
        if rank == 0:
            print(f"\n🔍 [滚动搜参] 区间: {val_start.date()} ~ {val_end.date()}")

        min_data_len = min(len(df) for df in val_data.values())
        max_lookback_allowed = min_data_len - self.config["pred_len"] - 5
        filtered_lookbacks = [lb for lb in param_space["lookback"] if lb <= max_lookback_allowed]
        if not filtered_lookbacks: filtered_lookbacks = [30]

        final_space = {"T": param_space["T"], "top_p": param_space["top_p"], "lookback": filtered_lookbacks}
        param_names = list(final_space.keys())
        all_combinations = list(product(*final_space.values()))
        
        # === 🔥 并行切分 ===
        my_combinations = all_combinations[rank::world_size]
        total_tasks = len(my_combinations)
        
        if rank == 0:
            print(f"   ⚙️ 总组合: {len(all_combinations)} | 显卡数: {world_size} | 单卡任务: ~{total_tasks}")

        local_best_ic = {name: -2.0 for name in val_data.keys()}
        local_best_params = {name: None for name in val_data.keys()}

        log_interval = max(1, total_tasks // 5) 

        import time
        start_time = time.time()

        for i, combo in enumerate(my_combinations):
            params = dict(zip(param_names, combo))
            scores_dict = self._evaluate_params_per_index(val_data, params, val_start, val_end)
            
            for name, score in scores_dict.items():
                if score > local_best_ic[name]:
                    local_best_ic[name] = score
                    local_best_params[name] = params.copy()
            
            if (i + 1) % log_interval == 0 or (i + 1) == total_tasks:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (total_tasks - (i + 1))

                print(f"   🚀 [GPU-{rank}] 进度 {i+1:2d}/{total_tasks} ({((i+1)/total_tasks)*100:.0f}%) | "
                      f"⏱️ {elapsed:.1f}s (剩 {remaining:.1f}s)")

        if rank == 0: print("   ⏳ 等待其他 GPU 完成任务...")
        
        my_result = (local_best_params, local_best_ic)
        all_results_list = [None for _ in range(world_size)]
        
        if world_size > 1:
            try:
                dist.all_gather_object(all_results_list, my_result)
            except Exception as e:
                print(f"❌ [Rank {rank}] Gather Error: {e}")
                all_results_list = [my_result]
        else:
            all_results_list = [my_result]

        # === 决出全局最优 (仅 Rank 0) ===
        if rank == 0:
            final_best_params = {name: None for name in val_data.keys()}
            final_best_ic = {name: -2.0 for name in val_data.keys()}

            for gpu_params, gpu_ics in all_results_list:
                if gpu_params is None: continue 
                for name in val_data.keys():
                    if gpu_ics[name] > final_best_ic[name]:
                        final_best_ic[name] = gpu_ics[name]
                        final_best_params[name] = gpu_params[name]
            
            print("\n   🏆 [全局汇总] 各指数最优参数:")
            default_params = {k: v[0] for k, v in final_space.items()}
            for name in val_data.keys():
                if final_best_params[name] is None:
                    final_best_params[name] = default_params
                else:
                    p = final_best_params[name]
                    ic = final_best_ic[name]
                    print(f"     ✅ {name}: IC={ic:.4f} (T={p['T']}, LB={p['lookback']})")
            
            return final_best_params
        
        return {}

    def _evaluate_params_per_index(self, val_data, params, val_start, val_end):
        lookback = params["lookback"]
        pred_len = self.config["pred_len"]
        results = {}

        for name, df in val_data.items():
            date_mask = (df["date"] >= val_start) & (df["date"] <= val_end)
            all_indices = df[date_mask].index.tolist()
            valid_indices = [i for i in all_indices if i <= (len(df) - 1 - pred_len) and i >= lookback]
            
            daily_sequence_corrs = []

            for idx in valid_indices:
                input_df = df.iloc[idx - lookback + 1 : idx + 1]
                future_dates = df.iloc[idx + 1 : idx + 1 + pred_len]["date"]
                
                pred_df = self.predictor.predict(
                    df=input_df, x_timestamp=input_df["date"],
                    y_timestamp=future_dates,
                    pred_len=pred_len, T=params["T"], top_p=params["top_p"],
                    sample_count=1, verbose=False
                )
                
                # 【关键修复】计算单次预测序列（长度为5）与真实序列的相关性
                actual_seq = df.iloc[idx + 1 : idx + 1 + pred_len]["close"].values
                pred_seq = pred_df["close"].values
                
                corr, _ = spearmanr(actual_seq, pred_seq)
                
                # 仅在计算出有效数值时统计
                if not np.isnan(corr):
                    daily_sequence_corrs.append(corr)

            # 取每日序列相关性的算术平均值作为该参数组在验证集上的得分
            results[name] = np.mean(daily_sequence_corrs)
                    
        return results

def run_rolling_system():
    print("="*60)
    print("🚀 微调版 Kronos 滚动择时系统 (最终拼接版 - 缓存优化 & 静默推理)")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/4] 加载全量测试数据...")
    all_data = read_test_data()
    all_data["时间"] = pd.to_datetime(all_data["时间"])
    
    # 2. 加载模型
    print("\n[2/4] 初始化模型...")
    try:
        tokenizer = KronosTokenizer.from_pretrained(CONFIG['tokenizer_path'])
        model = Kronos.from_pretrained(CONFIG['model_path'])
        predictor = KronosPredictor(
            model, tokenizer, device=CONFIG['device'], max_context=512,
            clip=CONFIG['clip_val']
        )
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return

    optimizer = ParameterOptimizer(predictor, CONFIG)
    
    # 滚动周期配置
    rolling_periods = [
        ("2025.01", "2024-12-01", "2024-12-31", "2025-01-01", "2025-01-31"),
        ("2025.02", "2025-01-01", "2025-01-31", "2025-02-01", "2025-02-28"),
        ("2025.03", "2025-02-01", "2025-02-28", "2025-03-01", "2025-03-31"),
        ("2025.04", "2025-03-01", "2025-03-31", "2025-04-01", "2025-04-30"),
        ("2025.05", "2025-04-01", "2025-04-30", "2025-05-01", "2025-05-31"),
        ("2025.06", "2025-05-01", "2025-05-31", "2025-06-01", "2025-06-30"),
        ("2025.07", "2025-06-01", "2025-06-30", "2025-07-01", "2025-07-31"),
        ("2025.08", "2025-07-01", "2025-07-31", "2025-08-01", "2025-08-31"),
        ("2025.09", "2025-08-01", "2025-08-31", "2025-09-01", "2025-09-30"),
    ]

    # === 初始化全局容器 ===
    all_metrics_buffer = [] 
    global_pred_buffers = {name: [] for name in INDICES.keys()}

    # === 🔵 缓存加载逻辑 ===
    param_cache = {}
    if rank == 0:
        if os.path.exists(PARAMS_CACHE_FILE):
            try:
                with open(PARAMS_CACHE_FILE, 'r') as f:
                    param_cache = json.load(f)
                print(f"📦 [Cache] 已加载参数缓存文件: {PARAMS_CACHE_FILE}")
            except Exception as e:
                print(f"⚠️ [Cache] 缓存文件读取失败: {e}")

    # 3. 滚动执行
    print("\n[3/4] 开始滚动推理...")
    
    for p_name, val_start, val_end, test_start, test_end in rolling_periods:
        if rank == 0:
            print(f"\n📅 周期: {p_name}")
        
        val_start_dt = pd.to_datetime(val_start)
        val_end_dt = pd.to_datetime(val_end)
        
        val_data_slice = {}
        max_lb = max(PARAM_SEARCH_SPACE['lookback'])
        history_buffer = pd.Timedelta(days=max_lb * 2) 
        
        for name, symbol in INDICES.items():
            df = all_data[all_data['代码'] == symbol].copy()
            mask = (df["时间"] >= (val_start_dt - history_buffer)) & (df["时间"] <= val_end_dt)
            slice_df = df[mask].rename(columns={
                "时间": "date", "开盘价(元)": "open", "最高价(元)": "high", 
                "最低价(元)": "low", "收盘价(元)": "close", "成交量(万股)": "volume"
            }).reset_index(drop=True)
            if not slice_df.empty:
                val_data_slice[name] = slice_df

        # --- A. 参数搜索 (带缓存检查) ---
        use_cache = False
        if rank == 0 and p_name in param_cache:
            use_cache = True

        decision_tensor = torch.tensor([1 if use_cache else 0], dtype=torch.int, device=CONFIG['device'])
        if world_size > 1:
            dist.broadcast(decision_tensor, src=0)
        should_use_cache = (decision_tensor.item() == 1)

        best_params_map = {}

        if should_use_cache:
            if rank == 0:
                print(f"   ⚡ [Cache] 命中缓存，跳过搜索")
                best_params_map = param_cache[p_name]
        else:
            best_params_map = optimizer.grid_search(val_data_slice, PARAM_SEARCH_SPACE, val_start_dt, val_end_dt)
            if rank == 0:
                param_cache[p_name] = best_params_map
                try:
                    with open(PARAMS_CACHE_FILE, 'w') as f:
                        json.dump(param_cache, f, indent=2, default=str)
                    print(f"   💾 [Cache] 参数已更新并保存")
                except Exception as e:
                    print(f"   ⚠️ [Cache] 保存失败: {e}")

        # 确保所有卡完成搜参
        if world_size > 1:
            dist.barrier()

        # 将最佳参数广播给所有卡，便于后续多卡推理
        if world_size > 1:
            obj_list = [best_params_map]
            dist.broadcast_object_list(obj_list, src=0)
            best_params_map = obj_list[0]

        # --- B. 样本外测试（多卡推理，搜参结束后统一进入） ---
        for name, symbol in INDICES.items():
            my_params = best_params_map.get(name, best_params_map.get(next(iter(best_params_map)) if best_params_map else name, {}))
            if not my_params:
                if rank == 0:
                    print(f"   ⚠️ {name} 无可用参数，跳过")
                continue

            idx_config = CONFIG.copy()
            idx_config.update(my_params)
            idx_config["test_start"] = test_start
            idx_config["test_end"] = test_end

            single_index_dict = {name: symbol}

            p_metrics, p_results = run_distributed_inference(
                predictor=predictor,
                all_data=all_data,
                indices_dict=single_index_dict,
                config=idx_config,
                output_dir=OUTPUT_DIR,
                model_name=f"temp_{p_name}_{name}",
                rank=rank,
                world_size=world_size,
            )

            if rank == 0:
                print(f"     📝 [Inference] {name} 完成 ({len(p_results.get(name, [])) if p_results else 0} days)", flush=True)

                for m in p_metrics:
                    m["period"] = p_name
                    m["best_T"] = my_params["T"]
                    m["best_LB"] = my_params["lookback"]
                all_metrics_buffer.extend(p_metrics)

                if name in p_results:
                    global_pred_buffers[name].append(p_results[name])

    # 4. 汇总、拼接与保存 (Rank 0 only)
    if rank == 0:
        print("\n[4/4] 汇总数据与绘图...")
        
        # === 拼接所有月份的预测结果 ===
        final_full_predictions = {}
        
        for name, df_list in global_pred_buffers.items():
            if df_list:
                full_df = pd.concat(df_list, axis=0).sort_values("date").reset_index(drop=True)
                final_full_predictions[name] = full_df
                
                save_path = os.path.join(OUTPUT_DIR, f"predictions_rolling_{name}.csv")
                full_df.to_csv(save_path, index=False)
                print(f"   💾 已保存完整预测: {save_path} (Rows: {len(full_df)})")
            else:
                print(f"   ⚠️ 未收集到指数 {name} 的预测数据")

        # === 分离明细保存与汇总展示 ===
        if all_metrics_buffer:
            print(f"   📊 正在合并 {len(all_metrics_buffer)} 个分片 Metrics...")
            
            # 1. 纵向拼接所有分片
            df_metrics = pd.concat(all_metrics_buffer, ignore_index=True)
            
            # 2. 保存明细
            detail_save_path = os.path.join(OUTPUT_DIR, "rolling_metrics_detailed.csv")
            df_metrics.to_csv(detail_save_path, index=False)
            print(f"   💾 已保存分月明细指标: {detail_save_path}")

            # 3. 计算平均
            df_avg = df_metrics.groupby(['Index', 'horizon']).mean(numeric_only=True).reset_index()
            
            aggregate_and_save_metrics([df_avg], OUTPUT_DIR, "rolling_all_avg")
            
        else:
            print("   ⚠️ Metrics Buffer 为空，跳过汇总")
        
        print(f"\n🎨 正在为 {len(final_full_predictions)} 个指数生成图表...")
        plot_all_results(final_full_predictions, OUTPUT_DIR, "rolling_final", CONFIG, combine_subplots=True)

        # 清理批次推理生成的临时预测文件
        temp_removed = 0
        for fname in os.listdir(OUTPUT_DIR):
            if fname.startswith("predictions_temp_") and fname.endswith(".csv"):
                temp_path = os.path.join(OUTPUT_DIR, fname)
                try:
                    os.remove(temp_path)
                    temp_removed += 1
                except Exception as e:
                    print(f"   ⚠️ 临时文件删除失败: {temp_path} ({e})")
        if temp_removed:
            print(f"   🧹 已清理 {temp_removed} 个临时预测文件")
        
        print(f"\n✅ 全部完成! 输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_rolling_system()