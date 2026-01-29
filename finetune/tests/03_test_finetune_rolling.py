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

CONFIG = FINETUNE_CONFIG | {
    "device": torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else "cpu"
}

PARAMS_CACHE_FILE = "/gemini/data-1/rolling_params_cache.json"
PARAM_SEARCH_SPACE = {
    "T": [0.3, 0.6, 0.8, 1.0],
    "top_p": [0.2, 0.4, 0.6, 0.9],
    "lookback": [30, 60, 90],
}

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "finetuned_rolling_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ParameterOptimizer:
    def __init__(self, predictor: KronosPredictor, config: Dict):
        self.predictor = predictor
        self.config = config

    def grid_search(
        self,
        val_data: Dict[str, pd.DataFrame],
        param_space: Dict,
        val_start: pd.Timestamp,
        val_end: pd.Timestamp,
    ) -> Dict[str, Dict]:
        if rank == 0:
            print(f"\n🔍 [滚动搜参] 区间: {val_start.date()} ~ {val_end.date()}")

        min_data_len = min(len(df) for df in val_data.values())
        max_lookback_allowed = min_data_len - self.config["pred_len"] - 5
        filtered_lookbacks = [
            lb for lb in param_space["lookback"] if lb <= max_lookback_allowed
        ]
        if not filtered_lookbacks:
            filtered_lookbacks = [30]

        final_space = {
            "T": param_space["T"],
            "top_p": param_space["top_p"],
            "lookback": filtered_lookbacks,
        }
        param_names = list(final_space.keys())
        all_combinations = list(product(*final_space.values()))

        # === 🔥 并行切分 ===
        my_combinations = all_combinations[rank::world_size]
        total_tasks = len(my_combinations)

        if rank == 0:
            print(
                f"   ⚙️ 总组合: {len(all_combinations)} | 显卡数: {world_size} | 单卡任务: ~{total_tasks}"
            )

        local_best_ic = {name: -2.0 for name in val_data.keys()}
        local_best_params = {name: None for name in val_data.keys()}

        log_interval = max(1, total_tasks // 5)

        import time

        start_time = time.time()

        for i, combo in enumerate(my_combinations):
            params = dict(zip(param_names, combo))
            scores_dict = self._evaluate_params_per_index(
                val_data, params, val_start, val_end
            )

            for name, score in scores_dict.items():
                if score > local_best_ic[name]:
                    local_best_ic[name] = score
                    local_best_params[name] = params.copy()

            if (i + 1) % log_interval == 0 or (i + 1) == total_tasks:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (total_tasks - (i + 1))

                print(
                    f"   🚀 [GPU-{rank}] 进度 {i+1:2d}/{total_tasks} ({((i+1)/total_tasks)*100:.0f}%) | "
                    f"⏱️ {elapsed:.1f}s (剩 {remaining:.1f}s)"
                )

        if rank == 0:
            print("   ⏳ 等待其他 GPU 完成任务...")

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
                if gpu_params is None:
                    continue
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
                    print(
                        f"     ✅ {name}: IC={ic:.4f} (T={p['T']},top_p ={p['top_p']} , LB={p['lookback']})"
                    )

            return final_best_params

        return {}

    def _evaluate_params_per_index(self, val_data: Dict[str, pd.DataFrame], params, val_start, val_end):
        lookback = params["lookback"]
        pred_len = self.config["pred_len"]
        results = {}

        for name, df in val_data.items():
            date_mask = (df["date"] >= val_start) & (df["date"] <= val_end)
            all_indices = df[date_mask].index.tolist()
            valid_indices = [i for i in all_indices if i <= (len(df) - 1 - pred_len) and i >= lookback]

            daily_sequence_corrs = []
            pred_t5_returns = []
            actual_t5_returns = []

            for idx in valid_indices:
                current_close = df.iloc[idx]["close"]
                input_df = df.iloc[idx - lookback + 1 : idx + 1]
                future_df = df.iloc[idx + 1 : idx + 1 + pred_len]
                
                pred_df = self.predictor.predict(
                    df=input_df, x_timestamp=input_df["date"],
                    y_timestamp=future_df["date"],
                    pred_len=pred_len, T=params["T"], top_p=params["top_p"],
                    sample_count=1, verbose=False,
                )

                # 1. 每日路径相关性 (含 T=0 锚点)
                actual_seq = np.insert(future_df["close"].values, 0, current_close)
                pred_seq = np.insert(pred_df["close"].values, 0, current_close)
                seq_corr, _ = spearmanr(actual_seq, pred_seq)
                daily_sequence_corrs.append(seq_corr)

                # 2. 收集全月 T+5 端点收益率
                p_ret_t5 = pred_df.iloc[pred_len - 1]["close"] / current_close - 1
                r_ret_t5 = future_df.iloc[pred_len - 1]["close"] / current_close - 1
                pred_t5_returns.append(p_ret_t5)
                actual_t5_returns.append(r_ret_t5)

            # 全月 T+5 择时收益率的相关性 (这是策略的核心)
            target_ic, _ = spearmanr(pred_t5_returns, actual_t5_returns)
            
            # 路径形状的平均相关性 (这是策略的辅助)
            mean_path_ic = np.mean(daily_sequence_corrs)

            # 【核心改进】将权重向 T+5 倾斜 (例如 0.8 / 0.2)
            # 这确保了搜参系统选出的参数首要保证 T+5 算得准，其次才是中间路径画得像
            results[name] = 0.8 * target_ic + 0.2 * mean_path_ic

        return results


def run_rolling_system():
    print("=" * 60)
    print("🚀 微调版 Kronos 滚动择时系统 (最终拼接版 - 缓存优化 & 静默推理)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载全量测试数据...")
    all_data = read_test_data()
    all_data["时间"] = pd.to_datetime(all_data["时间"])

    # 2. 加载模型
    print("\n[2/4] 初始化模型...")
    try:
        tokenizer = KronosTokenizer.from_pretrained(CONFIG["tokenizer_path"])
        model = Kronos.from_pretrained(CONFIG["model_path"])
        predictor = KronosPredictor(
            model,
            tokenizer,
            device=CONFIG["device"],
            max_context=512,
            clip=CONFIG["clip_val"],
        )
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return

    optimizer = ParameterOptimizer(predictor, CONFIG)

    # 动态生成滚动周期配置（基于 CONFIG 中的 test_start 和 test_end）
    from dateutil.relativedelta import relativedelta
    
    test_start_dt = pd.to_datetime(CONFIG["test_start"])
    test_end_dt = pd.to_datetime(CONFIG["test_end"])
    
    # 第一个验证期往前推一个月
    val_start_dt = test_start_dt - relativedelta(months=1)
    
    rolling_periods = []
    current_val_start = val_start_dt
    current_test_start = test_start_dt
    
    while current_test_start <= test_end_dt:
        # 计算当前月份的验证期和测试期
        current_val_end = current_test_start - pd.Timedelta(days=1)
        current_test_end = (current_test_start + relativedelta(months=1)) - pd.Timedelta(days=1)
        
        # 确保不超出总测试范围
        if current_test_end > test_end_dt:
            current_test_end = test_end_dt
        
        # 格式化为 YYYY.MM 的周期名
        period_name = current_test_start.strftime("%Y.%m")
        
        rolling_periods.append((
            period_name,
            current_val_start.strftime("%Y-%m-%d"),
            current_val_end.strftime("%Y-%m-%d"),
            current_test_start.strftime("%Y-%m-%d"),
            current_test_end.strftime("%Y-%m-%d"),
        ))
        
        # 移向下一个月
        current_val_start = current_test_start
        current_test_start = current_test_start + relativedelta(months=1)

    # === 初始化全局容器 ===
    global_pred_buffers = {name: [] for name in INDICES.keys()}

    # === 🔵 缓存加载逻辑 ===
    param_cache = {}
    if rank == 0:
        if os.path.exists(PARAMS_CACHE_FILE):
            try:
                with open(PARAMS_CACHE_FILE, "r") as f:
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
        max_lb = max(PARAM_SEARCH_SPACE["lookback"])
        history_buffer = pd.Timedelta(days=max_lb * 2)

        for name, symbol in INDICES.items():
            df = all_data[all_data["代码"] == symbol].copy()
            mask = (df["时间"] >= (val_start_dt - history_buffer)) & (
                df["时间"] <= val_end_dt
            )
            slice_df = (
                df[mask]
                .rename(
                    columns={
                        "时间": "date",
                        "开盘价(元)": "open",
                        "最高价(元)": "high",
                        "最低价(元)": "low",
                        "收盘价(元)": "close",
                        "成交量(万股)": "volume",
                    }
                )
                .reset_index(drop=True)
            )
            if not slice_df.empty:
                val_data_slice[name] = slice_df

        # --- A. 参数搜索 (带缓存检查) ---
        use_cache = False
        if rank == 0 and p_name in param_cache:
            use_cache = True

        decision_tensor = torch.tensor(
            [1 if use_cache else 0], dtype=torch.int, device=CONFIG["device"]
        )
        if world_size > 1:
            dist.broadcast(decision_tensor, src=0)
        should_use_cache = decision_tensor.item() == 1

        best_params_map = {}

        if should_use_cache:
            if rank == 0:
                print(f"   ⚡ [Cache] 命中缓存，跳过搜索")
                best_params_map = param_cache[p_name]
        else:
            best_params_map = optimizer.grid_search(
                val_data_slice, PARAM_SEARCH_SPACE, val_start_dt, val_end_dt
            )
            if rank == 0:
                param_cache[p_name] = best_params_map
                try:
                    with open(PARAMS_CACHE_FILE, "w") as f:
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
            my_params = best_params_map.get(
                name,
                best_params_map.get(
                    next(iter(best_params_map)) if best_params_map else name, {}
                ),
            )
            if not my_params:
                if rank == 0:
                    print(f"   ⚠️ {name} 无可用参数，跳过")
                continue

            idx_config = CONFIG.copy()
            idx_config.update(my_params)
            idx_config["test_start"] = test_start
            idx_config["test_end"] = test_end

            single_index_dict = {name: symbol}

            p_results = run_distributed_inference(
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
                print(
                    f"     📝 [Inference] {name} 完成 ({len(p_results.get(name, [])) if p_results else 0} days)",
                    flush=True,
                )

                if name in p_results:
                    global_pred_buffers[name].append(p_results[name])

    # 4. 汇总、拼接与全局指标计算 (Rank 0 only)
    if rank == 0:
        print("\n[4/4] 汇总全样本数据并计算全局指标...")

        final_full_predictions = {}
        global_metrics_list = []

        for name, df_list in global_pred_buffers.items():
            # 拼接全样本预测
            full_df = (
                pd.concat(df_list, axis=0).sort_values("date").reset_index(drop=True)
            )
            final_full_predictions[name] = full_df

            # 保存完整预测 CSV
            save_path = os.path.join(OUTPUT_DIR, f"predictions_rolling_{name}.csv")
            full_df.to_csv(save_path, index=False)

            # 【关键修改】直接在拼接后的全样本上调用 calculate_metrics
            # 这将计算包含跨月价格趋势的 Global IC，对齐报告 0.856 口径
            from testutils.metrics_utils import calculate_metrics

            idx_global_metrics = calculate_metrics(full_df)
            idx_global_metrics["Index"] = name
            global_metrics_list.append(idx_global_metrics)

            print(f"   ✅ {name}: 全样本 Global IC 汇总完成")

        # 保存并展示全局指标
        if global_metrics_list:
            df_global = pd.concat(global_metrics_list, ignore_index=True)

            # 专门保存一份对齐报告口径的汇总文件
            aggregate_and_save_metrics(df_global, OUTPUT_DIR, "rolling_global_final")

            print("\n🏆 全样本绝对价格相关性 (Global Price Corr):")
            t5_results = df_global[df_global["horizon"] == "T+5"]
            print(t5_results[["Index", "horizon", "price_corr", "price_mae"]])

        # 绘图逻辑保持不变
        plot_all_results(
            final_full_predictions,
            OUTPUT_DIR,
            "rolling_final",
            CONFIG,
            combine_subplots=True,
        )

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
