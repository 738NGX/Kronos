import os
import json
import numpy as np
import pandas as pd
import torch
from itertools import product
from typing import Dict, List, Tuple
from scipy.stats import spearmanr
import torch.distributed as dist
from dateutil.relativedelta import relativedelta
import heapq

# 环境初始化
from testutils.test_utils import (
    setup_environment, aggregate_and_save_metrics,
    init_distributed_mode, run_distributed_inference,
)
from testutils.common_config import FINETUNE_CONFIG, INDICES, BASE_OUTPUT_DIR
from testutils.data_utils import load_and_prepare_index_data, read_test_data

setup_environment()
from model import Kronos, KronosTokenizer, KronosPredictor

rank, local_rank, world_size = init_distributed_mode()

CONFIG = FINETUNE_CONFIG | {
    "device": torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else "cpu"
}

PARAMS_CACHE_FILE = "/gemini/data-1/rolling_params_cache.json"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "finetuned_rolling_test_v13_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAM_SEARCH_SPACE = {
    "T": [0.3, 0.6, 0.9, 1.2],
    "top_p": [0.3, 0.6, 0.9],
    "lookback": [30, 60, 90], 
}

class ParameterOptimizer:
    def __init__(self, predictor: KronosPredictor, config: Dict):
        self.predictor = predictor
        self.config = config

    def grid_search(self, val_data: Dict[str, pd.DataFrame], param_space: Dict, val_start: pd.Timestamp, val_end: pd.Timestamp, top_k: int = 3) -> Dict[str, List[Dict]]:
        """
        返回每个 Index 的 Top-K 参数组合列表
        """
        results = {}
        BATCH_SIZE = 64 

        for name, df in val_data.items():
            # 使用最小堆来维护 Top-K (score, params)，方便淘汰分低的
            top_k_heap = [] 
            
            val_mask = (df["date"] >= val_start) & (df["date"] <= val_end)
            if rank == 0:
                print(f"📊 [GPU{rank}-{name}] 审计区间: {val_start.date()} ~ {val_end.date()}")

            if df[val_mask].empty:
                raise ValueError(f"CRITICAL: Validation data empty for {name} between {val_start} and {val_end}")

            for lb in param_space["lookback"]:
                if lb > (len(df[val_mask]) - self.config["pred_len"]): 
                    continue

                v_idx = [i for i in df[val_mask].index if i <= (len(df) - 1 - self.config["pred_len"]) and i >= lb]
                
                if not v_idx:
                    continue

                for t, tp in product(param_space["T"], param_space["top_p"]):
                    
                    all_pred_rets, all_actual_rets = [], []
                    all_pred_prices, all_actual_prices = [], []

                    # --- Batch 处理防止 OOM ---
                    for i in range(0, len(v_idx), BATCH_SIZE):
                        batch_indices = v_idx[i : i + BATCH_SIZE]
                        
                        df_list_batch = [df.iloc[idx-lb+1 : idx+1].copy() for idx in batch_indices]
                        x_ts_batch = [df.iloc[idx-lb+1 : idx+1]["date"] for idx in batch_indices]
                        y_ts_batch = [df.iloc[idx+1 : idx+1+self.config["pred_len"]]["date"] for idx in batch_indices]

                        preds_batch = self.predictor.predict_batch(
                            df_list_batch, x_ts_batch, y_ts_batch, 
                            self.config["pred_len"], t, 0, tp, 10, False
                        )

                        for j, idx in enumerate(batch_indices):
                            c_now = df.iloc[idx]["close"]
                            if c_now == 0: raise ValueError(f"CRITICAL: Zero price at index {idx}")
                            
                            real_price = df.iloc[idx + self.config["pred_len"]]["close"]
                            pred_price = preds_batch[j].iloc[self.config["pred_len"] - 1]["close"]
                            
                            all_actual_rets.append(real_price / c_now - 1)
                            all_pred_rets.append(pred_price / c_now - 1)
                            all_actual_prices.append(real_price)
                            all_pred_prices.append(pred_price)
                        
                        del df_list_batch, preds_batch

                    # 指标计算
                    ic, _ = spearmanr(all_pred_rets, all_actual_rets)
                    if np.std(all_pred_prices) < 1e-9:
                         raise ValueError(f"CRITICAL: Model Collapse for {name} params {t}/{tp}/{lb}")
                    
                    p_corr = np.corrcoef(all_pred_prices, all_actual_prices)[0, 1]
                    quality_score = ic * 0.9 + p_corr * 0.1
                    
                    # 构造参数包
                    params_pack = {
                        "T": t, "top_p": tp, "lookback": lb, 
                        "ic": float(ic), "p_corr": float(p_corr), 
                        "score": quality_score
                    }

                    # --- 维护 Top-K 堆 ---
                    if len(top_k_heap) < top_k:
                        heapq.heappush(top_k_heap, (quality_score, params_pack))
                    else:
                        # 如果当前分数比堆里最小的还大，则替换
                        if quality_score > top_k_heap[0][0]:
                            heapq.heapreplace(top_k_heap, (quality_score, params_pack))

            if not top_k_heap:
                raise RuntimeError(f"❌ [GPU{rank}-{name}] 优化失败：未找到任何可行参数组合！")
            
            # 排序：从高分到低分
            sorted_top_k = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
            results[name] = [item[1] for item in sorted_top_k]
            
            best = results[name][0]
            print(f"✨ [GPU{rank}-{name}] Top1: Score={best['score']:.4f} | IC={best['ic']:.4f} | T={best['T']} | LB={best['lookback']}")

        return results

def run_rolling_system():
    # 1. 严格读取全量数据
    all_data = read_test_data() 

    predictors, optimizers = {}, {}
    for name in INDICES.keys():
        m_p = CONFIG['model_path'][name] if isinstance(CONFIG['model_path'], dict) else CONFIG['model_path']
        t_p = CONFIG['tokenizer_path'][name] if isinstance(CONFIG['tokenizer_path'], dict) else CONFIG['tokenizer_path']
        tokenizer = KronosTokenizer.from_pretrained(t_p)
        model = Kronos.from_pretrained(m_p)
        predictors[name] = KronosPredictor(model, tokenizer, device=CONFIG["device"], max_context=512, clip=CONFIG["clip_val"])
        optimizers[name] = ParameterOptimizer(predictors[name], CONFIG)

    curr = pd.to_datetime(CONFIG["test_start"])
    test_end_dt = pd.to_datetime(CONFIG["test_end"])
    rolling_periods = []
    while curr <= test_end_dt:
        v_s, v_e = curr - relativedelta(months=3), curr - pd.Timedelta(days=1)
        t_e = min(curr + relativedelta(months=1) - pd.Timedelta(days=1), test_end_dt)
        rolling_periods.append((curr.strftime("%Y.%m"), v_s.strftime("%Y-%m-%d"), v_e.strftime("%Y-%m-%d"), curr.strftime("%Y-%m-%d"), t_e.strftime("%Y-%m-%d")))
        curr += relativedelta(months=1)

    global_pred_buffers = {name: {} for name in INDICES.keys()} # 改为字典：name -> rank -> list
    param_cache = {}
    
    # 缓存读取
    if rank == 0 and os.path.exists(PARAMS_CACHE_FILE):
        print(f"📂 [GPU{rank}] 加载缓存: {PARAMS_CACHE_FILE}")
        with open(PARAMS_CACHE_FILE, "r") as f: param_cache = json.load(f)
    
    cache_object = [param_cache]
    dist.broadcast_object_list(cache_object, src=0)
    param_cache = cache_object[0]

    for p_name, v_s, v_e, t_s, t_e in rolling_periods:
        if rank == 0: print(f"\n▶️ 周期: {p_name} | 测试区间: {t_s} ~ {t_e}")
        
        current_top_k_params = {} # 结构: {name: [param1, param2, param3]}
        skip_optimization = False

        # --- 缓存检查 ---
        if p_name in param_cache:
            # 简单校验 keys
            if set(INDICES.keys()).issubset(set(param_cache[p_name].keys())):
                if rank == 0: print(f"⏩ [Cache Hit] 周期 {p_name} 命中。")
                current_top_k_params = param_cache[p_name]
                skip_optimization = True

        # --- 执行搜索 ---
        if not skip_optimization:
            v_s_dt, v_e_dt = pd.to_datetime(v_s), pd.to_datetime(v_e)
            val_data_slice = {}
            for name, symbol in INDICES.items():
                temp_config = {"test_start": t_s, "test_end": t_e}
                processed_df, _ = load_and_prepare_index_data(all_data, name, symbol, temp_config)
                mask = (processed_df["date"] >= (v_s_dt - pd.Timedelta(days=120))) & (processed_df["date"] <= v_e_dt)
                slice_df = processed_df[mask].sort_values("date").reset_index(drop=True)
                if len(slice_df) < 30: raise ValueError(f"CRITICAL: Insufficient validation data for {name}")
                val_data_slice[name] = slice_df

            local_results = {}
            idx_list = list(INDICES.keys())
            for name in idx_list[rank::world_size]:
                # 这里的 grid_search 已经改为返回 list
                local_results.update(optimizers[name].grid_search({name: val_data_slice[name]}, PARAM_SEARCH_SPACE, v_s_dt, v_e_dt, top_k=3))
            
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_results)
            for d in gathered: 
                if d: current_top_k_params.update(d)
            
            if len(current_top_k_params) != len(INDICES):
                raise RuntimeError("CRITICAL: Optimization incomplete.")

            if rank == 0:
                param_cache[p_name] = current_top_k_params
                with open(PARAMS_CACHE_FILE, "w") as f: json.dump(param_cache, f, indent=2, default=str)
        
        dist.barrier()

        # --- 推理执行 (遍历 Top 3) ---
        for name, symbol in INDICES.items():
            top_params_list = current_top_k_params[name]
            
            # 遍历 Top K 参数，分别进行推理
            for rank_idx, params in enumerate(top_params_list):
                rank_id = rank_idx + 1 # 1, 2, 3
                p_config = CONFIG | params | {"test_start": t_s, "test_end": t_e}
                
                # 临时文件名带上 rank 标识
                temp_file_suffix = f"temp_{p_name}_{name}_rank{rank_id}"
                
                p_res = run_distributed_inference(
                    predictors[name], all_data, {name: symbol}, p_config, 
                    OUTPUT_DIR, temp_file_suffix, rank, world_size
                )
                
                if rank == 0: 
                    if name not in p_res or p_res[name].empty:
                         raise RuntimeError(f"Inference empty for {name} rank {rank_id}")
                    
                    # 初始化存储结构
                    if rank_id not in global_pred_buffers[name]:
                        global_pred_buffers[name][rank_id] = []
                    global_pred_buffers[name][rank_id].append(p_res[name])
        
        dist.barrier()

    # 4. 结果聚合与报告
    if rank == 0:
        print("\n🧹 生成最终报告 (Top-K Mode)...")
        from testutils.metrics_utils import calculate_metrics
        final_metrics = []
        
        # 遍历每个 Index 的每个 Rank
        for name, rank_dict in global_pred_buffers.items():
            for rank_id, df_list in rank_dict.items():
                if not df_list: continue
                
                full_df = pd.concat(df_list).sort_values("date").reset_index(drop=True)
                
                # 保存原始预测文件
                save_name = f"final_predictions_{name}_rank{rank_id}.csv"
                full_df.to_csv(os.path.join(OUTPUT_DIR, save_name), index=False)
                
                # 计算指标
                res_metrics = calculate_metrics(full_df)
                # 兼容性处理：转字典
                m_dict = res_metrics.to_dict() if hasattr(res_metrics, 'to_dict') else dict(res_metrics)
                m_dict["Index"] = name
                m_dict["Rank"] = rank_id 
                final_metrics.append(m_dict)
        
        # 将所有指标转为 DataFrame
        metrics_df = pd.DataFrame(final_metrics)
        
        # --- 修复点：按 Rank 拆分报告，避免 pivot 重复索引报错 ---
        unique_ranks = sorted(metrics_df["Rank"].unique())
        
        for r_id in unique_ranks:
            print(f"\n📊 生成 Rank {r_id} 报告...")
            # 筛选当前 Rank 的数据
            sub_df = metrics_df[metrics_df["Rank"] == r_id].copy()
            # 移除 Rank 列，防止干扰工具函数
            if "Rank" in sub_df.columns:
                sub_df = sub_df.drop(columns=["Rank"])
            
            # 分别调用工具函数保存报告
            try:
                aggregate_and_save_metrics(sub_df, OUTPUT_DIR, f"v14_topk_rank{r_id}")
            except Exception as e:
                print(f"⚠️ 生成 Rank {r_id} 报告时工具函数报错 (非致命): {e}")
                # 手动备份一份 CSV 以防工具函数完全失败
                sub_df.to_csv(os.path.join(OUTPUT_DIR, f"backup_metrics_rank{r_id}.csv"), index=False)

        # 清理临时文件
        print("\n🧹 清理临时碎片...")
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith("predictions_temp_"):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except OSError:
                    pass
            
        print("🏁 全流程成功结束。所有 Top-3 结果已按 Rank 分组保存。")

if __name__ == "__main__":
    run_rolling_system()

if __name__ == "__main__":
    run_rolling_system()