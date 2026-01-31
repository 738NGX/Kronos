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
        all_indices_names = list(val_data.keys())
        my_indices = all_indices_names[rank::world_size]
        
        if rank == 0:
            print(f"\n🔍 [滚动搜参-Batch高性能版] 区间: {val_start.date()} ~ {val_end.date()}")
            print(f"   策略：全量采样 (Step=1) + 批量并行推理")

        local_best_score = {name: -2.0 for name in all_indices_names}
        local_best_params = {name: None for name in all_indices_names}

        for name in my_indices:
            df = val_data[name]
            min_data_len = len(df)
            max_lookback_allowed = min_data_len - self.config["pred_len"] - 5
            filtered_lookbacks = [lb for lb in param_space["lookback"] if lb <= max_lookback_allowed]
            
            # --- 优化：将 Lookback 设为外层循环，利用 Batch 提速 ---
            for lb in (filtered_lookbacks if filtered_lookbacks else [30]):
                date_mask = (df["date"] >= val_start) & (df["date"] <= val_end)
                all_idx = df[date_mask].index.tolist()
                valid_indices = [i for i in all_idx if i <= (len(df) - 1 - self.config["pred_len"]) and i >= lb]

                # 1. 构造该 Lookback 下的全量 Batch
                df_list, x_ts_list, y_ts_list = [], [], []
                for idx in valid_indices:
                    df_list.append(df.iloc[idx - lb + 1 : idx + 1])
                    x_ts_list.append(df.iloc[idx - lb + 1 : idx + 1]["date"])
                    y_ts_list.append(df.iloc[idx + 1 : idx + 1 + self.config["pred_len"]]["date"])

                # 2. 调用官方 Batch 接口 (全量推理 90 天仅需极少次数的模型 Forward)
                # 为保证多样性评估，这里先以基准参数获取 Batch 预测
                import time
                batch_start = time.time()
                
                # 注意：这里我们遍历 T 和 top_p。如果模型推理开销极大，
                # 我们可以根据官方文档建议，通过调整 predict_batch 的参数来遍历组合
                for t, tp in product(param_space["T"], param_space["top_p"]):
                    pred_df_list = self.predictor.predict_batch(
                        df_list=df_list,
                        x_timestamp_list=x_ts_list,
                        y_timestamp_list=y_ts_list,
                        pred_len=self.config["pred_len"],
                        T=t,
                        top_p=tp,
                        sample_count=10,
                        verbose=False
                    )

                    # 3. 内存评估 (计算 Hybrid Score)
                    score = self._evaluate_batch_results(df, valid_indices, pred_df_list)

                    if score > local_best_score[name]:
                        local_best_score[name] = score
                        local_best_params[name] = {"T": t, "top_p": tp, "lookback": lb}

                if rank == 0:
                    elapsed = time.time() - batch_start
                    print(f"   🚀 [GPU-{rank}] 指数: {name} | LB={lb} 完成评估 | 耗时: {elapsed:.1f}s")

        # === 汇总结果 (保持同步) ===
        my_result = (local_best_params, local_best_score)
        all_results_list = [None for _ in range(world_size)]
        if world_size > 1:
            dist.all_gather_object(all_results_list, my_result)
        else:
            all_results_list = [my_result]

        final_best_params = {name: None for name in all_indices_names}
        final_best_score = {name: -2.0 for name in all_indices_names}

        for gpu_params, gpu_scores in all_results_list:
            for name in all_indices_names:
                if gpu_params[name] is not None:
                    if gpu_scores[name] > final_best_score[name]:
                        final_best_score[name] = gpu_scores[name]
                        final_best_params[name] = gpu_params[name]

        if rank == 0:
            print("\n🏆 [全局汇总] 各指数最优参数 (Hybrid Score):")
            for name in val_data.keys():
                p = final_best_params[name]
                print(f"   ✅ {name}: Score={final_best_score[name]:.4f} (T={p['T']}, top_p={p['top_p']}, LB={p['lookback']})")

        return final_best_params

    def _evaluate_batch_results(self, df, indices, pred_df_list):
        """
        利用预计算的 Batch 结果进行快速策略模拟
        """
        pred_len = self.config["pred_len"]
        strategy_daily_returns = []
        pred_t5_returns = []
        actual_t5_returns = []
        holding_timer = 0
        
        for i, idx in enumerate(indices):
            current_close = df.iloc[idx]["close"]
            market_ret_next = df.iloc[idx + 1]["close"] / current_close - 1
            
            # 从 Batch 结果中提取对应的预测
            pred_df = pred_df_list[i]
            
            # 计算 T+5 收益率
            p_ret_t5 = pred_df.iloc[pred_len - 1]["close"] / current_close - 1
            # 真实收益率 (从原始 df 获取以确保准确)
            r_ret_t5 = df.iloc[idx + pred_len]["close"] / current_close - 1
            
            pred_t5_returns.append(p_ret_t5)
            actual_t5_returns.append(r_ret_t5)

            # 择时逻辑模拟
            if holding_timer == 0 and p_ret_t5 > 0.005:
                holding_timer = 5
            
            if holding_timer > 0:
                strategy_daily_returns.append(market_ret_next)
                holding_timer -= 1
            else:
                strategy_daily_returns.append(0.0)

        # 指标计算
        ic, _ = spearmanr(pred_t5_returns, actual_t5_returns)
        ret_array = np.array(strategy_daily_returns)
        
        if np.std(ret_array) > 0:
            sharpe = (np.mean(ret_array) / np.std(ret_array)) * np.sqrt(252)
        else:
            sharpe = -1.0
            
        return 0.5 * ic + 0.5 * (sharpe / 2.0)


def run_rolling_system():
    print("=" * 60)
    print("🚀 微调版 Kronos 滚动择时系统 (最终拼接版 - 缓存优化 & 静默推理)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载全量测试数据...")
    all_data = read_test_data()
    all_data["时间"] = pd.to_datetime(all_data["时间"])

    # 2. 加载模型（支持每个指数使用不同的模型）
    print("\n[2/4] 初始化模型...")
    tokenizer = KronosTokenizer.from_pretrained(CONFIG["tokenizer_path"])
    
    # 存储每个指数的预测器和优化器
    predictors = {}
    optimizers = {}
    
    for index in INDICES.keys():
        try:
            # 获取每个指数对应的模型路径
            if isinstance(CONFIG['model_path'], dict):
                model_path = CONFIG['model_path'].get(index, CONFIG['model_path'].get('default', list(CONFIG['model_path'].values())[0]))
            else:
                model_path = CONFIG['model_path']  # 如果是字符串，所有指数使用同一个模型
            
            if rank == 0:
                print(f"   加载 {index} 的模型: {model_path}")
            
            model = Kronos.from_pretrained(model_path)
            predictor = KronosPredictor(
                model,
                tokenizer,
                device=CONFIG["device"],
                max_context=512,
                clip=CONFIG["clip_val"],
            )
            predictors[index] = predictor
            optimizers[index] = ParameterOptimizer(predictor, CONFIG)
            
            if rank == 0:
                print(f"   ✅ {index} 模型加载成功")
        except Exception as e:
            print(f"   ❌ {index} 模型加载失败: {e}")
            return

    # 动态生成滚动周期配置（基于 CONFIG 中的 test_start 和 test_end）
    from dateutil.relativedelta import relativedelta
    
    test_start_dt = pd.to_datetime(CONFIG["test_start"])
    test_end_dt = pd.to_datetime(CONFIG["test_end"])
    
    # 修改点：第一个验证期往前推 3 个月
    val_start_dt = test_start_dt - relativedelta(months=3)
    
    rolling_periods = []
    current_test_start = test_start_dt
    
    while current_test_start <= test_end_dt:
        # 核心逻辑：测试当月，验证过去三个月
        current_val_start = current_test_start - relativedelta(months=3)
        current_val_end = current_test_start - pd.Timedelta(days=1)
        current_test_end = (current_test_start + relativedelta(months=1)) - pd.Timedelta(days=1)
        
        if current_test_end > test_end_dt:
            current_test_end = test_end_dt
        
        period_name = current_test_start.strftime("%Y.%m")
        
        rolling_periods.append((
            period_name,
            current_val_start.strftime("%Y-%m-%d"),
            current_val_end.strftime("%Y-%m-%d"),
            current_test_start.strftime("%Y-%m-%d"),
            current_test_end.strftime("%Y-%m-%d"),
        ))
        
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
            # 为每个指数使用对应的优化器进行参数搜索
            for index in INDICES.keys():
                if index in optimizers and index in val_data_slice:
                    index_params = optimizers[index].grid_search(
                        {index: val_data_slice[index]}, PARAM_SEARCH_SPACE, val_start_dt, val_end_dt
                    )
                    best_params_map.update(index_params)
            
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

            # 使用该指数对应的预测器
            index_predictor = predictors.get(name, list(predictors.values())[0])
            
            p_results = run_distributed_inference(
                predictor=index_predictor,
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

            from testutils.metrics_utils import calculate_metrics

            idx_global_metrics = calculate_metrics(full_df)
            idx_global_metrics["Index"] = name
            global_metrics_list.append(idx_global_metrics)

            print(f"   ✅ {name}: 全样本 Global IC 汇总完成")

        # 保存并展示全局指标
        if global_metrics_list:
            df_global = pd.concat(global_metrics_list, ignore_index=True)

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
