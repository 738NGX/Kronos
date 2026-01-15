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

warnings.filterwarnings('ignore')
sys.path.append("/gemini/code/")

# ================= 0. 实验配置 =================
class Config:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
    model_path = "NeoQuasar/Kronos-base"
    train_dir = "/gemini/code/dataset/train"
    target_index_file = "/gemini/code/dataset/inference/000852.SH.csv"
    results_dir = "/gemini/code/results_final_pass"
    
    cache_file = "/gemini/code/dataset/train_cache_aligned_final.pt"
    map_file = "/gemini/code/dataset/vocab_map_final.pt"
    
    lookback = 96
    pred_len = 5
    epochs = 6
    batch_size = 32
    lr = 5e-5
    weight_decay = 1e-4
    sample_count = 10
    threshold = 0.003
    seed = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

try:
    from model import Kronos, KronosTokenizer
    print("✅ [System] Loaded Kronos Modules")
    USE_MOCK = False
except ImportError:
    print("⚠️ Mock Mode")
    USE_MOCK = True
    class KronosTokenizer(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, x): return cls()
        def encode(self, x): return torch.randint(50000, 100000, (x.shape[0], x.shape[1], 2)).long().to(x.device)
        def decode(self, x): return torch.randn(x.shape[0], x.shape[1], 6).to(x.device)
        def to(self, d): return self
    class RealEmbeddingWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embeddings = nn.Embedding(1024, 768)
        def forward(self, inputs): return self.token_embeddings(inputs[0])
    class Kronos(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, x): return cls()
        def __init__(self):
            super().__init__()
            self.embedding = RealEmbeddingWrapper()
            self.layers = nn.ModuleList([nn.Linear(768, 768) for _ in range(2)])
            self.head = nn.Linear(768, 1024)
            # 模拟问题根源：固化的属性
            self.s1_vocab_size = 1024 
        def forward(self, x, s2_ids=None):
            h = self.embedding([x, s2_ids])
            h = self.layers[0](h)
            logits = self.head(h)
            # 模拟报错行：使用 self.s1_vocab_size 进行 view
            # 如果不更新 s1_vocab_size，这里必然报错
            try:
                _ = logits.view(-1, self.s1_vocab_size)
            except: pass # 只是模拟
            return type('O', (), {'logits': logits})()
        def to(self, d): return self
        def train(self): pass
        def parameters(self): return [torch.randn(1, requires_grad=True)]

# ================= 1. 数据预处理 =================
def preprocess_and_align(data_dir, tokenizer):
    if os.path.exists(Config.cache_file) and os.path.exists(Config.map_file):
        try: return torch.load(Config.cache_file), torch.load(Config.map_file)
        except: pass
    
    print(f"⚙️ 启动预处理 (Report Standard)...")
    if not USE_MOCK: tokenizer.to(Config.device)
    
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    files = [f for f in files if Config.target_index_file not in f][:3000]
    
    raw_token_batches = []
    total_seq_len = Config.lookback + Config.pred_len
    batch_buffer = []
    
    for f in tqdm(files, desc="Tokenizing"):
        try:
            df = pd.read_csv(f)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['close'])
            if len(df) < total_seq_len + 10: continue
            if 'amount' not in df.columns: df['amount'] = df['close'] * df['volume']
            
            raw_close = df['close'].values.astype(np.float64)
            if np.any(raw_close <= 0): continue
            returns = np.diff(raw_close) / raw_close[:-1]
            returns = np.insert(returns, 0, 0.0)
            mask = (returns < -0.3) | (returns > 0.3)
            returns[mask] = 0.0
            adj_close = np.cumprod(1 + returns)
            factor = adj_close / raw_close
            
            df['close'] = adj_close
            df['open'] *= factor; df['high'] *= factor; df['low'] *= factor; df['volume'] /= factor
            
            if df.isnull().values.any(): df = df.fillna(method='ffill').fillna(0)
            arr = df[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)
            arr = np.nan_to_num(arr)
            
            max_start = len(arr) - total_seq_len
            starts = np.random.randint(0, max_start, min(20, max_start))
            
            for start in starts:
                seq = arr[start : start + total_seq_len]
                mean = np.mean(seq, axis=0, keepdims=True)
                std = np.std(seq, axis=0, keepdims=True) + 1e-6
                norm_seq = (seq - mean) / std
                norm_seq = np.nan_to_num(norm_seq)
                batch_buffer.append(norm_seq)
                
                if len(batch_buffer) >= 256:
                    tensor_in = torch.tensor(np.array(batch_buffer)).to(Config.device)
                    with torch.no_grad(): tokens = tokenizer.encode(tensor_in)
                    if tokens.dim() == 2: tokens = tokens.unsqueeze(-1).repeat(1, 1, 2)
                    raw_token_batches.append(tokens.cpu())
                    batch_buffer = []
        except Exception: continue
    
    if batch_buffer:
        tensor_in = torch.tensor(np.array(batch_buffer)).to(Config.device)
        with torch.no_grad(): tokens = tokenizer.encode(tensor_in)
        if tokens.dim() == 2: tokens = tokens.unsqueeze(-1).repeat(1, 1, 2)
        raw_token_batches.append(tokens.cpu())
    
    if not raw_token_batches: return torch.empty(0), {}
    
    all_raw_tokens = torch.cat(raw_token_batches, dim=0)
    unique_tokens = torch.unique(all_raw_tokens).sort()[0]
    reverse_map = unique_tokens 
    raw_flat = all_raw_tokens.numpy().flatten()
    unique_np = unique_tokens.numpy()
    mapped_indices = np.searchsorted(unique_np, raw_flat)
    mapped_data = torch.tensor(mapped_indices).reshape(all_raw_tokens.shape).long()
    
    torch.save(mapped_data, Config.cache_file)
    torch.save(reverse_map, Config.map_file)
    return mapped_data, reverse_map

# ================= 2. 安全模型调整 (Safe Resize + Attr Sync) =================
def resize_model_safely(model, target_vocab_size):
    print(f"\n🔧 [Safe Resize] Adapting model to vocab size: {target_vocab_size}")
    
    old_vocab = 1024 # Default
    hidden_dim = 768 # Default
    
    # 1. 侦测配置
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            if m.num_embeddings > 256: 
                old_vocab = m.num_embeddings
                hidden_dim = m.embedding_dim
                print(f"   ℹ️ Detected Old Config: Vocab={old_vocab}, Hidden={hidden_dim}")
                break

    # 2. Resize Embedding (Leaf Nodes)
    for name, m in model.named_modules():
        if isinstance(m, nn.Embedding):
            if m.num_embeddings == old_vocab and m.embedding_dim == hidden_dim:
                new_emb = nn.Embedding(target_vocab_size, hidden_dim).to(Config.device)
                
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                if '.' not in name: setattr(model, name, new_emb)
                else:
                    parent = model
                    for part in parent_name.split('.'): parent = getattr(parent, part)
                    setattr(parent, child_name, new_emb)
                
                print(f"   ✅ Resized Input Embedding: {name}")
    
    # 3. Resize Output Head
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if m.in_features == hidden_dim and m.out_features == old_vocab:
                new_head = nn.Linear(hidden_dim, target_vocab_size).to(Config.device)
                
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                if '.' not in name: setattr(model, name, new_head)
                else:
                    parent = model
                    for part in parent_name.split('.'): parent = getattr(parent, part)
                    setattr(parent, child_name, new_head)
                    
                print(f"   ✅ Resized Output Head: {name}")

    # --- 🔥 关键修复：同步更新模型属性 ---
    # 防止 forward 函数中使用旧的 self.s1_vocab_size 导致 view 报错
    print("   🔄 Syncing model attributes...")
    
    # 更新 Config (如果有)
    if hasattr(model, 'config'):
        if hasattr(model.config, 'vocab_size'): model.config.vocab_size = target_vocab_size
        if hasattr(model.config, 's1_vocab_size'): model.config.s1_vocab_size = target_vocab_size
        if hasattr(model.config, 's2_vocab_size'): model.config.s2_vocab_size = target_vocab_size
    
    # 更新 Model 自身属性 (如果有)
    attrs_to_check = ['vocab_size', 's1_vocab_size', 's2_vocab_size', 'n_vocab', 'vocab']
    for attr in attrs_to_check:
        if hasattr(model, attr):
            # 只有当属性值等于 old_vocab 时才更新，防止误伤其他参数
            val = getattr(model, attr)
            if isinstance(val, int) and val == old_vocab:
                setattr(model, attr, target_vocab_size)
                print(f"      -> Updated model.{attr} to {target_vocab_size}")

    return model

# ================= 3. 微调与回测 =================
class CausalDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def finetune(model, tokenizer):
    full_data, reverse_map = preprocess_and_align(Config.train_dir, tokenizer)
    if len(full_data) == 0: return model, reverse_map
    
    vocab_needed = len(reverse_map)
    model = resize_model_safely(model, vocab_needed)
    
    print(f"\n🚀 [Step 1] 微调 (Vocab={vocab_needed})...")
    dataset = CausalDataset(full_data)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    optimizer = AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    if not USE_MOCK: model.train()
    
    for epoch in range(Config.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            batch = batch.to(Config.device)
            if batch.dim() == 2: batch = batch.unsqueeze(-1).repeat(1, 1, 2)
            
            s1_ids, s2_ids = batch[:, :, 0], batch[:, :, 1]
            inp_s1, inp_s2 = s1_ids[:, :-1], s2_ids[:, :-1]
            targets = s1_ids[:, 1:]
            
            optimizer.zero_grad()
            output = model(inp_s1, s2_ids=inp_s2)
            logits = output.logits if hasattr(output, 'logits') else (output[0] if isinstance(output, tuple) else output)
            
            shift_logits = logits[:, -Config.pred_len:, :].reshape(-1, vocab_needed)
            shift_labels = targets[:, -Config.pred_len:].reshape(-1)
            
            loss = criterion(shift_logits, shift_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    return model, reverse_map

class InferenceDataset(Dataset):
    def __init__(self, df, lookback, pred_len):
        self.df = df; self.lookback = lookback; self.pred_len = pred_len
        self.indices = range(lookback, len(df) - pred_len, 1)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        window_df = self.df.iloc[idx - self.lookback : idx]
        raw_vals = window_df[['open', 'high', 'low', 'close', 'volume', 'amount']].values.astype(np.float32)
        mean = np.mean(raw_vals, axis=0, keepdims=True)
        std = np.std(raw_vals, axis=0, keepdims=True) + 1e-6
        norm_vals = np.nan_to_num((raw_vals - mean) / std)
        return torch.tensor(norm_vals), mean[0, 3], std[0, 3], window_df['close'].iloc[-1], self.df['close'].iloc[idx + self.pred_len], str(window_df.index[-1])

def run_backtest(model, tokenizer, reverse_map):
    print("\n🚀 [Step 2] 启动滚动回测...")
    if not USE_MOCK: tokenizer.to(Config.device)
    reverse_map = reverse_map.to(Config.device)
    
    df = pd.read_csv(Config.target_index_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    if 'amount' not in df.columns: df['amount'] = df['close'] * df['volume']
    
    test_len = 300
    if len(df) > test_len + Config.lookback: df = df.iloc[-(test_len + Config.lookback + Config.pred_len + 10):]
    loader = DataLoader(InferenceDataset(df, Config.lookback, Config.pred_len), batch_size=Config.batch_size, shuffle=False)
    
    model.eval()
    results = []
    
    for inputs, means, stds, curr_prices, future_prices, timestamps in tqdm(loader, desc="Infer"):
        B = inputs.shape[0]
        inputs = inputs.to(Config.device)
        try:
            with torch.no_grad():
                if not USE_MOCK: raw_tokens = tokenizer.encode(inputs)
                else: raw_tokens = torch.randint(50000, 52000, (B, Config.lookback, 2)).to(Config.device)
                if raw_tokens.dim() == 2: raw_tokens = raw_tokens.unsqueeze(-1).repeat(1, 1, 2)
                
                flat_raw = raw_tokens.flatten()
                flat_raw = torch.clamp(flat_raw, min=reverse_map[0], max=reverse_map[-1])
                mapped_indices = torch.searchsorted(reverse_map, flat_raw)
                mapped_indices = torch.clamp(mapped_indices, 0, len(reverse_map) - 1)
                dense_tokens = mapped_indices.reshape(raw_tokens.shape)
                
                K = Config.sample_count
                batch_s1 = dense_tokens[:, :, 0].repeat_interleave(K, dim=0)
                batch_s2 = dense_tokens[:, :, 1].repeat_interleave(K, dim=0)
                
                future_ids = []
                for _ in range(Config.pred_len):
                    output = model(batch_s1, s2_ids=batch_s2)
                    logits = output.logits if hasattr(output, 'logits') else (output[0] if isinstance(output, tuple) else output)
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    next_dense = torch.multinomial(probs, 1)
                    future_ids.append(next_dense)
                    batch_s1 = torch.cat([batch_s1, next_dense], dim=1)
                    batch_s2 = torch.cat([batch_s2, batch_s2[:, -1:]], dim=1)
                
                pred_dense = torch.cat(future_ids, dim=1)
                pred_raw = reverse_map[pred_dense]
                
                if not USE_MOCK: pred_vals = tokenizer.decode(pred_raw)
                else: pred_vals = pred_raw.float().unsqueeze(-1).repeat(1,1,6)
                
                avg_z = torch.mean(pred_vals[:, :, 3][:, -1].view(B, K), dim=1).cpu()
                pred_prices = avg_z * stds + means
                
                for i in range(B):
                    p_pred, p_curr, p_fut = pred_prices[i].item(), curr_prices[i].item(), future_prices[i].item()
                    results.append({'timestamp': timestamps[i], 'close': p_curr, 'signal': 1 if (p_pred/p_curr - 1) > Config.threshold else 0, 'actual_ret': (p_fut/p_curr - 1)})
        except Exception as e: continue

    if results:
        res_df = pd.DataFrame(results).set_index(pd.to_datetime([r['timestamp'] for r in results])).sort_index()
        res_df['strat_ret'] = res_df['signal'] * res_df['actual_ret']
        res_df['nav'] = (1 + res_df['strat_ret']).cumprod()
        res_df['bench'] = (1 + res_df['actual_ret']).cumprod()
        os.makedirs(Config.results_dir, exist_ok=True)
        res_df.to_csv(os.path.join(Config.results_dir, "report_pass.csv"))
        sharpe = (res_df['strat_ret'].mean() * 252) / (res_df['strat_ret'].std() * np.sqrt(252/Config.pred_len) + 1e-9)
        plt.figure(figsize=(10, 5))
        plt.plot(res_df.index, res_df['nav'], label=f'Kronos (Sharpe: {sharpe:.2f})', color='#d62728')
        plt.plot(res_df.index, res_df['bench'], label='Benchmark', color='grey', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(Config.results_dir, "report_pass.png"))
        print(f"\n✅ 运行完成 | NAV: {res_df['nav'].iloc[-1]:.4f}")

if __name__ == "__main__":
    set_seed(Config.seed)
    tokenizer = KronosTokenizer.from_pretrained(Config.tokenizer_path)
    model = Kronos.from_pretrained(Config.model_path).to(Config.device)
    model, rmap = finetune(model, tokenizer)
    run_backtest(model, tokenizer, rmap)