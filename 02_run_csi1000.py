import os
import glob
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import warnings

# 禁用警告
warnings.filterwarnings('ignore')
sys.path.append("/gemini/code/")

# ================= 0. 实验配置 (Report Standard) =================
class Config:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
    model_path = "NeoQuasar/Kronos-base"
    
    train_dir = "/gemini/code/dataset/train"
    target_index_file = "/gemini/code/dataset/inference/000852.SH.csv"
    results_dir = "/gemini/code/results_report_final"
    
    # 缓存设定
    cache_file = "/gemini/code/dataset/train_cache_cs_final.pt"
    map_file = "/gemini/code/dataset/vocab_map_cs_final.pt"
    model_save_file = "/gemini/code/dataset/kronos_tuned_final.pth" # 🔥 救命存档文件
    
    lookback = 96
    pred_len = 5
    epochs = 3
    batch_size = 32
    lr = 5e-5
    
    top_k = 50
    rebalance_days = 5
    seed = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from model import Kronos, KronosTokenizer

# ================= 1. 物理结构手术 (保持不变) =================
def resize_kronos_internal(model, target_vocab_size):
    print(f"🔧 [Surgery] 物理同步 Kronos 结构: Vocab -> {target_vocab_size}")
    hidden_dim = 832 
    model.embedding.emb_s1 = nn.Embedding(target_vocab_size, hidden_dim).to(Config.device)
    model.embedding.emb_s2 = nn.Embedding(target_vocab_size, hidden_dim).to(Config.device)
    model.head.proj_s1 = nn.Linear(hidden_dim, target_vocab_size).to(Config.device)
    model.head.proj_s2 = nn.Linear(hidden_dim, target_vocab_size).to(Config.device)
    model.s1_vocab_size = target_vocab_size
    model.s2_vocab_size = target_vocab_size
    return model

# ================= 2. 数据处理 (保持不变) =================
def get_data(tokenizer):
    if os.path.exists(Config.cache_file):
        print("📦 [Cache] 加载数据缓存...")
        return torch.load(Config.cache_file), torch.load(Config.map_file)
    
    print("⚙️ [Processing] 开始全量数据处理...")
    tokenizer.to(Config.device)
    files = [f for f in glob.glob(os.path.join(Config.train_dir, "*.csv")) if "000852" not in f]
    
    all_tokens = []
    batch_buffer = []
    for f in tqdm(files, desc="Parsing CSVs"):
        try:
            df = pd.read_csv(f).dropna(subset=['close'])
            if len(df) < 110: continue
            raw = df['close'].values.astype(np.float64)
            df['close'] = np.cumprod(1 + np.insert(np.diff(raw)/raw[:-1], 0, 0))
            arr = df[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)
            
            for start in range(0, len(arr) - 101, 5):
                seq = arr[start : start + 101]
                batch_buffer.append((seq - np.mean(seq, axis=0)) / (np.std(seq, axis=0) + 1e-6))
                
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

    full_raw = torch.cat(all_tokens, dim=0)
    rev_map = torch.unique(full_raw).sort()[0]
    mapped = torch.tensor(np.searchsorted(rev_map.numpy(), full_raw.numpy())).reshape(full_raw.shape).long()
    
    torch.save(mapped, Config.cache_file)
    torch.save(rev_map, Config.map_file)
    return mapped, rev_map

# ================= 3. 微调 (带自动存档功能) =================
def run_train(model, tokenizer):
    data, rmap = get_data(tokenizer)
    model = resize_kronos_internal(model, len(rmap))
    
    # 🔥 核心修正：如果存在权重存档，直接加载，不再训练！
    if os.path.exists(Config.model_save_file):
        print(f"💾 [Checkpoint] 检测到已训练模型: {Config.model_save_file}")
        print("⏩ 跳过训练步骤，直接加载权重...")
        model.load_state_dict(torch.load(Config.model_save_file))
        return model, rmap
    
    print("🚀 [Training] 未检测到存档，开始 4.5h 训练...")
    loader = DataLoader(data, batch_size=Config.batch_size, shuffle=True)
    opt = AdamW(model.parameters(), lr=Config.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for e in range(Config.epochs):
        pbar = tqdm(loader, desc=f"Epoch {e+1}")
        for b in pbar:
            b = b.to(Config.device)
            out = model(b[:, :-1, 0], s2_ids=b[:, :-1, 1])
            logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, 'logits') else out)
            loss = loss_fn(logits[:, -5:, :].reshape(-1, len(rmap)), b[:, -5:, 0].reshape(-1))
            loss.backward(); opt.step(); opt.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    # 🔥 训练完立刻保存，防止白跑
    print(f"💾 [Save] 训练完成，正在保存模型至 {Config.model_save_file}...")
    torch.save(model.state_dict(), Config.model_save_file)
    return model, rmap

# ================= 4. 截面回测 (修正日期逻辑) =================
def run_backtest(model, tokenizer, rmap):
    print("\n🚀 [Alpha] 启动截面选股回测 (修正版)...")
    model.eval(); rmap = rmap.to(Config.device)
    
    idx_df = pd.read_csv(Config.target_index_file)
    idx_df['timestamp'] = pd.to_datetime(idx_df['timestamp']).dt.normalize()
    idx_df = idx_df.set_index('timestamp').sort_index()
    
    stocks = {}
    files = glob.glob(os.path.join(Config.train_dir, "*.csv"))
    # 加载前 1500 个以测试 (或者全量)
    for f in tqdm(files, desc="Loading Pool"):
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
            stocks[os.path.basename(f).split('.')[0]] = df.set_index('timestamp').sort_index()
        except: continue

    # --- 🔥 关键逻辑修正：以指数时间为准 (Master Clock) ---
    # 不要去求所有股票的交集，那会得到空集。
    # 我们只关心：在指数存在的那些日子里，有哪些股票是活着的。
    
    # 取指数最后 500 个交易日作为测试区间 (约2年)
    trading_dates = idx_df.index[-500:]
    # 每5天调仓
    rebalance_dates = trading_dates[::5]
    
    print(f"📅 回测区间: {rebalance_dates[0].date()} 至 {rebalance_dates[-1].date()}")
    print(f"📊 调仓次数: {len(rebalance_dates)}")
    
    results = []
    
    for d in tqdm(rebalance_dates[:-1], desc="Rebalancing"):
        batch_x, codes = [], []
        
        # 遍历所有股票，检查当天 d 是否有数据
        for c, df in stocks.items():
            if d not in df.index: continue # 当天停牌或未上市/已退市，跳过
            
            pos = df.index.get_loc(d)
            if pos < 96: continue # 上市时间太短，不够 lookback
            
            # 截取窗口
            win = df.iloc[pos-95 : pos+1][['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)
            # 简单的 Z-Score
            batch_x.append((win - np.mean(win, axis=0)) / (np.std(win, axis=0) + 1e-6))
            codes.append(c)
        
        # 如果当天一只股票都没有 (基本不可能，除非数据源全是错的)
        if not batch_x: 
            # print(f"⚠️ {d.date()} 无股票数据")
            continue
            
        with torch.no_grad():
            tensor_in = torch.tensor(np.array(batch_x)).to(Config.device)
            # Tokenize
            toks = tokenizer.encode(tensor_in)
            if toks.dim() == 2: toks = toks.unsqueeze(-1).repeat(1, 1, 2)
            
            # Align
            dense = torch.searchsorted(rmap, torch.clamp(toks, rmap[0], rmap[-1])).reshape(toks.shape)
            dense = torch.clamp(dense, 0, len(rmap)-1)
            
            # Predict
            out = model(dense[:, :, 0], s2_ids=dense[:, :, 1])
            logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, 'logits') else out)
            
            # Score: 取 Close (idx 3) 的预测值
            score = tokenizer.decode(rmap[torch.argmax(logits[:, -1, :], dim=-1)])[:, 3].cpu().numpy()

        # 选股
        rank = pd.DataFrame({'c': codes, 's': score}).sort_values('s', ascending=False)
        top = rank.head(50)['c'].tolist()
        
        # 计算未来 5 天收益
        rets = []
        for c in top:
            s_df = stocks[c]
            try:
                p = s_df.index.get_loc(d)
                # 买入价: d 的收盘价 (简化处理)
                # 卖出价: d+5 的收盘价
                idx_sell = min(p+5, len(s_df)-1)
                r = s_df['close'].iloc[idx_sell] / s_df['close'].iloc[p] - 1
                rets.append(r)
            except: pass
        
        # 计算指数同期收益
        try:
            b_p = idx_df.index.get_loc(d)
            b_idx_sell = min(b_p+5, len(idx_df)-1)
            b_ret = idx_df['close'].iloc[b_idx_sell] / idx_df['close'].iloc[b_p] - 1
        except: b_ret = 0.0
        
        results.append({'d': d, 'r': np.mean(rets) if rets else 0.0, 'b': b_ret})
        
        # 清理显存，防止 OOM
        torch.cuda.empty_cache()

    if results:
        res = pd.DataFrame(results).set_index('d')
        res['ex'] = res['r'] - res['b']
        res['nav'] = (1 + res['ex']).cumprod()
        res['bench'] = (1 + res['b']).cumprod()
        
        ir = (res['ex'].mean() / (res['ex'].std() + 1e-9)) * np.sqrt(252/5)
        
        plt.figure(figsize=(10, 5))
        plt.plot(res.index, res['nav'], label=f'Alpha (IR: {ir:.2f})', color='red')
        plt.plot(res.index, res['bench'], label='Benchmark', color='grey', alpha=0.5)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(Config.results_dir, "alpha_final.png"))
        print(f"\n✅ 运行完成 | IR: {ir:.4f} | 累计超额: {res['nav'].iloc[-1]:.4f}")
    else:
        print("\n❌ 依然没有结果。请检查指数 CSV 和个股 CSV 的日期格式是否完全一致 (YYYY-MM-DD)。")

if __name__ == "__main__":
    set_seed(Config.seed)
    tk = KronosTokenizer.from_pretrained(Config.tokenizer_path)
    md = Kronos.from_pretrained(Config.model_path).to(Config.device)
    md, rm = run_train(md, tk)
    run_backtest(md, tk, rm)