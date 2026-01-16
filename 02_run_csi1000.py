import os
import glob
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import warnings

# 禁用警告
warnings.filterwarnings('ignore')
sys.path.append("/gemini/code/")

# ================= 0. 实验配置 (严格对齐 Report Page 15/16) =================
class Config:
    device = "cuda:0"
    tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
    model_path = "NeoQuasar/Kronos-base"
    
    # 路径
    train_dir = "/gemini/code/dataset/train"
    target_index_file = "/gemini/code/dataset/inference/000852.SH.csv"
    results_dir = "/gemini/code/results_report_strict"
    
    # 缓存 (Lookback改变，必须重生成)
    cache_file = "/gemini/code/dataset/train_cache_250_strict.pt" 
    map_file = "/gemini/code/dataset/vocab_map_250_strict.pt"
    
    # --- 🔥 图19 参数复刻 🔥 ---
    lookback_window = 250   # 核心差异：看过去一年
    pred_window = 5         # 预测未来一周
    epochs = 10             # 训练轮数
    batch_size = 30         # 严格对齐报告
    n_train_iter = 60000    # 单轮采样数
    
    # 优化器参数
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    lr = 5e-5               # 报告未明示，取经验值
    weight_decay = 0.1      # 图19指定
    
    # 回测参数
    test_days = 500         # 留最后2年做样本外滚动测试
    seed = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from model import Kronos, KronosTokenizer

# ================= 1. 物理结构手术 =================
def resize_model_physical(model, target_vocab_size):
    print(f"🔧 [Surgery] 物理同步模型结构: Vocab -> {target_vocab_size}")
    hidden_dim = 832 
    
    # 直接修改内存中的模块
    model.embedding.emb_s1 = nn.Embedding(target_vocab_size, hidden_dim).to(Config.device)
    model.embedding.emb_s2 = nn.Embedding(target_vocab_size, hidden_dim).to(Config.device)
    model.head.proj_s1 = nn.Linear(hidden_dim, target_vocab_size).to(Config.device)
    model.head.proj_s2 = nn.Linear(hidden_dim, target_vocab_size).to(Config.device)
    
    # 同步控制属性
    model.s1_vocab_size = target_vocab_size
    model.s2_vocab_size = target_vocab_size
    return model

# ================= 2. 数据处理：构建全市场样本池 =================
def preprocess_universe_pool(tokenizer):
    if os.path.exists(Config.cache_file):
        print("📦 [Cache] 加载样本池缓存...")
        return torch.load(Config.cache_file), torch.load(Config.map_file)
    
    print(f"⚙️ [Processing] 启动全市场样本池构建 (Lookback={Config.lookback_window})...")
    tokenizer.to(Config.device)
    
    # 读取所有个股 (600+)
    files = glob.glob(os.path.join(Config.train_dir, "*.csv"))
    files = [f for f in files if "000852" not in f]
    print(f"📚 扫描股票文件数: {len(files)}")
    
    all_tokens = []
    batch_buffer = []
    
    # 序列总长 = 输入(250) + 输出(5)
    total_len = Config.lookback_window + Config.pred_window 
    
    for f in tqdm(files, desc="Scanning"):
        try:
            df = pd.read_csv(f).dropna(subset=['close'])
            if len(df) < total_len + 10: continue
            
            # 复权
            raw = df['close'].values.astype(np.float64)
            df['close'] = np.cumprod(1 + np.insert(np.diff(raw)/raw[:-1], 0, 0))
            arr = df[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)
            
            max_start = len(arr) - total_len
            
            # --- 🔥 关键策略：步长设为 5 ---
            # 600只股票 * 3000天 / 5 = 360,000 个样本
            # 足够覆盖报告要求的 60,000 iter，且不会爆内存
            for start in range(0, max_start, 5):
                seq = arr[start : start + total_len]
                # Rolling Z-Score
                mean = np.mean(seq, axis=0, keepdims=True)
                std = np.std(seq, axis=0, keepdims=True) + 1e-6
                norm_seq = (seq - mean) / std
                batch_buffer.append(np.nan_to_num(norm_seq))
                
                if len(batch_buffer) >= 512:
                    with torch.no_grad():
                        t = tokenizer.encode(torch.tensor(np.array(batch_buffer)).to(Config.device))
                    if t.dim() == 2: t = t.unsqueeze(-1).repeat(1, 1, 2)
                    all_tokens.append(t.cpu())
                    batch_buffer = []
        except: continue
        
    if batch_buffer:
        with torch.no_grad():
            t = tokenizer.encode(torch.tensor(np.array(batch_buffer)).to(Config.device))
        if t.dim() == 2: t = t.unsqueeze(-1).repeat(1, 1, 2)
        all_tokens.append(t.cpu())

    if not all_tokens: raise ValueError("❌ 样本池为空！")

    print(f"   📊 合并样本池...")
    full_raw = torch.cat(all_tokens, dim=0)
    print(f"   ✅ 样本池总容量: {len(full_raw)} (满足 n_train_iter={Config.n_train_iter})")
    
    rev_map = torch.unique(full_raw).sort()[0]
    mapped = torch.tensor(np.searchsorted(rev_map.numpy(), full_raw.numpy())).reshape(full_raw.shape).long()
    
    torch.save(mapped, Config.cache_file)
    torch.save(rev_map, Config.map_file)
    return mapped, rev_map

# ================= 3. 随机采样微调 (Report Method) =================
class RandomSampleDataset(Dataset):
    def __init__(self, data, samples_per_epoch):
        self.data = data
        self.samples_per_epoch = samples_per_epoch
        self.total = len(data)
    def __len__(self): return self.samples_per_epoch
    def __getitem__(self, idx):
        # 随机有放回采样
        return self.data[np.random.randint(0, self.total)]

def run_train(model, tokenizer):
    data, rmap = preprocess_universe_pool(tokenizer)
    vocab_needed = len(rmap)
    model = resize_model_physical(model, vocab_needed)
    
    # 构建符合报告采样逻辑的数据集
    train_dataset = RandomSampleDataset(data, Config.n_train_iter)
    
    print(f"\n🚀 [Training] 启动微调 (Report Page 16 Parameters)...")
    print(f"   Batch: {Config.batch_size} | Epochs: {Config.epochs} | Samples/Epoch: {Config.n_train_iter}")
    
    loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    # 指定 Betas 和 Weight Decay
    opt = Adam(model.parameters(), lr=Config.lr, betas=(Config.adam_beta1, Config.adam_beta2), weight_decay=Config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for e in range(Config.epochs):
        pbar = tqdm(loader, desc=f"Epoch {e+1}/{Config.epochs}")
        for b in pbar:
            b = b.to(Config.device)
            # Tuple 兼容处理
            out = model(b[:, :-1, 0], s2_ids=b[:, :-1, 1])
            logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, 'logits') else out)
            
            loss = loss_fn(logits[:, -Config.pred_window:, :].reshape(-1, vocab_needed), 
                           b[:, -Config.pred_window:, 0].reshape(-1))
            
            loss.backward(); opt.step(); opt.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    return model, rmap

# ================= 4. 滚动择时推演 (Rolling Inference) =================
def run_timing_inference(model, tokenizer, rmap):
    print("\n🚀 [Inference] 启动指数滚动择时 (Page 20)...")
    model.eval(); rmap = rmap.to(Config.device); tokenizer.to(Config.device)
    
    # 只读取指数文件
    df = pd.read_csv(Config.target_index_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 复权
    raw = df['close'].values.astype(np.float64)
    df['close'] = np.cumprod(1 + np.insert(np.diff(raw)/raw[:-1], 0, 0))
    arr = df[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)
    
    results = []
    # 滚动窗口推演：从倒数第 test_days 天开始
    start_idx = max(0, len(df) - Config.test_days)
    # 步长 = pred_window (5天)，避免信号重叠
    indices = range(start_idx, len(df) - (Config.lookback_window + Config.pred_window), Config.pred_window)
    
    print(f"📅 推演区间: {df['timestamp'].iloc[start_idx]} -> {df['timestamp'].iloc[-1]}")
    
    for i in tqdm(indices, desc="Rolling"):
        # 1. 截取历史 [T-250 : T]
        # 注意：这里 seq 长度是 250，用于预测未来
        input_seq = arr[i : i + Config.lookback_window]
        
        # 2. 实时 Z-Score
        mean = np.mean(input_seq, axis=0, keepdims=True)
        std = np.std(input_seq, axis=0, keepdims=True) + 1e-6
        norm_in = (input_seq - mean) / std
        
        # 3. 预测
        with torch.no_grad():
            t_in = torch.tensor(norm_in).unsqueeze(0).to(Config.device)
            toks = tokenizer.encode(t_in)
            if toks.dim() == 2: toks = toks.unsqueeze(-1).repeat(1, 1, 2)
            
            flat = toks.flatten()
            flat = torch.clamp(flat, min=rmap[0], max=rmap[-1])
            dense = torch.searchsorted(rmap, flat).reshape(toks.shape)
            dense = torch.clamp(dense, 0, len(rmap)-1)
            
            out = model(dense[:, :, 0], s2_ids=dense[:, :, 1])
            logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, 'logits') else out)
            
            # 取最后一步 (T时刻) 的预测输出
            best_id = torch.argmax(logits[:, -1, :], dim=-1)
            pred_val = tokenizer.decode(rmap[best_id])[:, 3].cpu().item() # Close Z-Score
            
        # 4. 交易逻辑
        # 预测 Z-Score > 0 (强于均值) -> 做多
        signal = 1 if pred_val > 0 else 0
        
        # 5. 结算
        p_buy = df['close'].iloc[i + Config.lookback_window - 1] # T时刻收盘价
        p_sell = df['close'].iloc[i + Config.lookback_window + Config.pred_window - 1] # T+5时刻收盘价
        ret = (p_sell / p_buy) - 1
        
        results.append({
            'date': df['timestamp'].iloc[i + Config.lookback_window - 1],
            'strat_ret': ret * signal,
            'bench_ret': ret
        })

    # 统计
    if results:
        res = pd.DataFrame(results).set_index('date')
        res['nav'] = (1 + res['strat_ret']).cumprod()
        res['bench'] = (1 + res['bench_ret']).cumprod()
        
        ann_ret = res['strat_ret'].mean() * (252 / Config.pred_window)
        ann_std = res['strat_ret'].std() * np.sqrt(252 / Config.pred_window)
        sharpe = (ann_ret - 0.02) / (ann_std + 1e-9)
        
        os.makedirs(Config.results_dir, exist_ok=True)
        res.to_csv(os.path.join(Config.results_dir, "timing_result.csv"))
        
        plt.figure(figsize=(10, 6))
        plt.plot(res.index, res['nav'], label=f'Kronos Timing (Sharpe: {sharpe:.2f})', color='red')
        plt.plot(res.index, res['bench'], label='Index Benchmark', color='grey', alpha=0.5)
        plt.title(f"Report Page 15 Replication (Lookback={Config.lookback_window})")
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join(Config.results_dir, "timing_curve.png"))
        
        print(f"\n✅ 复现完成 | Sharpe: {sharpe:.2f} | 净值: {res['nav'].iloc[-1]:.4f}")

if __name__ == "__main__":
    set_seed(Config.seed)
    tk = KronosTokenizer.from_pretrained(Config.tokenizer_path)
    md = Kronos.from_pretrained(Config.model_path).to(Config.device)
    md, rm = run_train(md, tk)
    run_timing_inference(md, tk, rm)