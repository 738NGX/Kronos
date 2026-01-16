import yfinance as yf
import pandas as pd
import os

# ================= 配置区域 =================
# 保存路径 (直接存到你的微调数据目录)
OUTPUT_DIR = r"./finetune_csv/data/global_gold"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "GLOBAL_GOLD.csv")

# 目标标的：COMEX 黄金期货 (24小时交易，包含美盘)
# 如果网络不好，也可以改用 "GLD" (美股黄金ETF)
TICKER = "GC=F" 

def download_and_clean_gold():
    print(f"🚀 开始下载国际黄金数据: {TICKER} ...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    try:
        # 1. 下载数据 (下载历史上所有数据)
        # auto_adjust=True 会自动处理复权
        df = yf.download(TICKER, period="max", progress=False, auto_adjust=True)
        
        if df.empty:
            print("❌ 下载失败：数据为空。请检查网络 (可能需要代理)。")
            return

        print(f"📥 原始数据下载成功: {len(df)} 条")

        # 2. 格式清洗
        # yfinance 的索引是 Date，列名是 Open, High, Low, Close, Volume
        df = df.reset_index()
        
        # 扁平化列名 (处理 MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 重命名为 Kronos 标准格式
        df = df.rename(columns={
            "Date": "timestamps",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        # 3. 核心清洗：去除非交易日和坏数据
        # 确保日期格式
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        # 只要核心列
        df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume']]
        
        # 过滤掉 Volume=0 的日子 (死盘/节假日)
        # 注意：COMEX黄金有时候Volume是0但价格在变，这里稍微宽容一点
        # 如果是 ETF (GLD)，Volume=0 通常意味着休市，可以删
        # 既然是微调，我们希望数据质量极高，删掉 Volume=0 是安全的
        original_len = len(df)
        df = df[df['volume'] > 0].dropna()
        print(f"🧹 清洗无效数据: 剔除 {original_len - len(df)} 行 (休市/坏点)")

        # 4. 排序
        df = df.sort_values('timestamps')
        
        # 计算并增加amount列（成交额）
        df['amount'] = df['close'] * df['volume']
        
        # 5. 保存
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ 成功！清洗后的黄金数据已保存至:\n   {OUTPUT_FILE}")
        print(f"   数据范围: {df['timestamps'].iloc[0].date()} 到 {df['timestamps'].iloc[-1].date()}")
        print(f"   总条数: {len(df)}")
        
        # 打印最后5行看看
        print("\n数据预览 (Last 5 rows):")
        print(df.tail())

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    download_and_clean_gold()