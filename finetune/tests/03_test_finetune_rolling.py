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
import time

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
        disable_rank_split: bool = False,
    ) -> Dict[str, Dict]:
        all_indices_names = list(val_data.keys())
        my_indices = all_indices_names if disable_rank_split else all_indices_names[rank::world_size]
        
        # 性能剪枝阈值：如果当前 lookback 已达到优秀水平，不再搜索更长的窗口
        PERFORMANCE_PRUNING_THRESHOLD = 1.3 

        local_best_score = {name: -2.0 for name in all_indices_names}
        local_best_params = {name: None for name in all_indices_names}

        for name in my_indices:
            df = val_data[name]
            min_data_len = len(df)
            # 这里的 5 是对应预测序列长度 pred_window
            max_lookback_allowed = min_data_len - self.config["pred_len"] - 5
            filtered_lookbacks = sorted([lb for lb in param_space["lookback"] if lb <= max_lookback_allowed])
            
            for lb in (filtered_lookbacks if filtered_lookbacks else [30]):
                # 剪枝：若已搜到足够好的参数，跳过后续更长的 Lookback 以节省 3.5 小时的总耗时
                if local_best_score[name] >= PERFORMANCE_PRUNING_THRESHOLD:
                    continue

                date_mask = (df["date"] >= val_start) & (df["date"] <= val_end)
                all_idx = df[date_mask].index.tolist()
                valid_indices = [i for i in all_idx if i <= (len(df) - 1 - self.config["pred_len"]) and i >= lb]

                # 1. 构造该 Lookback 下的全量 Batch
                df_list, x_ts_list, y_ts_list = [], [], []
                for idx in valid_indices:
                    df_list.append(df.iloc[idx - lb + 1 : idx + 1])
                    x_ts_list.append(df.iloc[idx - lb + 1 : idx + 1]["date"])
                    y_ts_list.append(df.iloc[idx + 1 : idx + 1 + self.config["pred_len"]]["date"])

                batch_start = time.time()
                
                # 2. 遍历参数组合进行批量推理
                for t, tp in product(param_space["T"], param_space["top_p"]):
                    # 采样次数固定为 30 以确保统计稳健性
                    pred_df_list = self.predictor.predict_batch(
                        df_list=df_list,
                        x_timestamp_list=x_ts_list,
                        y_timestamp_list=y_ts_list,
                        pred_len=self.config["pred_len"],
                        T=t,
                        top_p=tp,
                        sample_count=15,
                        verbose=False
                    )

                    # 3. 增强版评估：引入 Participation Rate 和 Recency Weighting
                    score = self._evaluate_batch_results_v2(df, valid_indices, pred_df_list)

                    if score > local_best_score[name]:
                        local_best_score[name] = score
                        local_best_params[name] = {"T": t, "top_p": tp, "lookback": lb}

                elapsed = time.time() - batch_start
                print(f" 🚀 [GPU-{rank}] 指数: {name} | LB={lb} 完成评估 | 耗时: {elapsed:.1f}s")

        # === 结果汇总逻辑 ===
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
                if name in gpu_params and gpu_params[name] is not None:
                    if gpu_scores[name] > final_best_score[name]:
                        final_best_score[name] = gpu_scores[name]
                        final_best_params[name] = gpu_params[name]

        if rank == 0:
            print("\n🏆 [全局汇总] 各指数最优参数 (Hybrid Score v2):")
            for name in val_data.keys():
                p = final_best_params[name]
                print(f"   ✅ {name}: Score={final_best_score[name]:.4f} (T={p['T']}, top_p={p['top_p']}, LB={p['lookback']})")

        return final_best_params

    def _evaluate_batch_results_v2(self, df, indices, pred_df_list):
        """
        评估函数 v2：旨在解决暴涨行情踏空问题
        """
        pred_len = self.config["pred_len"]
        strategy_daily_returns = []
        market_daily_returns = []
        pred_t5_returns = []
        actual_t5_returns = []
        positions = []
        holding_timer = 0
        
        # 1. 构造近期加权序列：近期样本权重更高，半衰期设为 20 个交易日
        num_samples = len(indices)
        time_weights = np.exp(np.linspace(-1.0, 0, num_samples)) 

        for i, idx in enumerate(indices):
            current_close = df.iloc[idx]["close"]
            mkt_ret_next = df.iloc[idx + 1]["close"] / current_close - 1
            market_daily_returns.append(mkt_ret_next)
            
            # 提取 T+5 预测与真实值
            pred_df = pred_df_list[i]
            p_ret_t5 = pred_df.iloc[pred_len - 1]["close"] / current_close - 1
            r_ret_t5 = df.iloc[idx + pred_len]["close"] / current_close - 1
            
            pred_t5_returns.append(p_ret_t5)
            actual_t5_returns.append(r_ret_t5)

            # 严格执行 0.5% 阈值与 5 天持仓择时
            if holding_timer == 0 and p_ret_t5 > 0.005:
                holding_timer = 5
            
            if holding_timer > 0:
                strategy_daily_returns.append(mkt_ret_next)
                positions.append(1.0)
                holding_timer -= 1
            else:
                strategy_daily_returns.append(0.0)
                positions.append(0.0)

        # 2. 计算加权 IC (体现预测的时效准确性)
        ic, _ = spearmanr(np.array(pred_t5_returns) * time_weights, 
                          np.array(actual_t5_returns) * time_weights)
        
        # 3. 计算加权夏普率
        strat_rets = np.array(strategy_daily_returns)
        weighted_strat_rets = strat_rets * time_weights
        if np.std(weighted_strat_rets) > 0:
            sharpe = (np.mean(weighted_strat_rets) / np.std(weighted_strat_rets)) * np.sqrt(252)
        else:
            sharpe = -1.0

        # 4. 计算 Beta 参与率得分 (Participation Score)
        # 核心逻辑：在验证期内指数表现最好的 20% 的日子里，策略的持仓比例
        mkt_array = np.array(market_daily_returns)
        top_20_threshold = np.percentile(mkt_array, 80)
        top_days_mask = mkt_array >= top_20_threshold
        
        participation_rate = np.mean(np.array(positions)[top_days_mask])

        # 5. 最终复合得分 2.0
        # 权重分配：30% IC + 30% Sharpe + 40% Participation
        # 此公式强制搜参系统在大涨行情（Participation）和稳定性（Sharpe）之间寻找平衡
        hybrid_score = 0.3 * ic + 0.3 * (sharpe / 2.0) + 0.4 * participation_rate
        
        return hybrid_score


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

    # 存储每个指数的预测器和优化器
    predictors = {}
    optimizers = {}

    # 缓存（按路径复用）
    tokenizer_cache = {}
    model_cache = {}
    
    for index in INDICES.keys():
        try:
            # 获取每个指数对应的模型路径
            if isinstance(CONFIG['model_path'], dict):
                model_path = CONFIG['model_path'].get(index, CONFIG['model_path'].get('default', list(CONFIG['model_path'].values())[0]))
            else:
                model_path = CONFIG['model_path']  # 如果是字符串，所有指数使用同一个模型
            
            if rank == 0:
                print(f"   加载 {index} 的模型: {model_path}")
            
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
            # 为每个指数使用对应的优化器进行参数搜索（按 rank 分配指数）
            all_indices = list(INDICES.keys())
            my_indices = all_indices[rank::world_size] if world_size > 1 else all_indices
            for index in my_indices:
                if index in optimizers and index in val_data_slice:
                    index_params = optimizers[index].grid_search(
                        {index: val_data_slice[index]},
                        PARAM_SEARCH_SPACE,
                        val_start_dt,
                        val_end_dt,
                        disable_rank_split=True,
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
