import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ================= 配置区域 =================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r"D:\Downloads\111" 

INDICES = {
    "上证50": "000016.SH",
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
    "中证2000": "932000.CSI",
    "中证红利": "000922.CSI",
    "恒生指数": "HSI.HK",
    "恒生科技": "HSTECH.HK",
    "黄金ETF": "518880.SH",
}


def _compute_metrics(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    metrics = {
        "price_corr": [],
        "price_mae": [],
        "ret_corr": [],
        "ret_mae": [],
    }
    current_close = df["current_close"].astype(float)
    for h in horizons:
        pred = df[f"pred_t+{h}"].astype(float)
        real = df[f"real_t+{h}"].astype(float)
        price_corr = pred.corr(real)
        price_mae = (pred - real).abs().mean()
        pred_ret = (pred - current_close) / current_close
        real_ret = (real - current_close) / current_close
        ret_corr = pred_ret.corr(real_ret)
        ret_mae = (pred_ret - real_ret).abs().mean()
        metrics["price_corr"].append(price_corr)
        metrics["price_mae"].append(price_mae)
        metrics["ret_corr"].append(ret_corr)
        metrics["ret_mae"].append(ret_mae)

    return pd.DataFrame(
        metrics,
        index=[f"T+{h}" for h in horizons],
    ).T


def _plot_file(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    horizons = [1, 2, 3, 4, 5]
    metrics_df = _compute_metrics(df, horizons)

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.2])

    axes = []
    for i in range(5):
        r, c = divmod(i, 2)
        axes.append(fig.add_subplot(gs[r, c]))

    for ax, h in zip(axes, horizons):
        pred_col = f"pred_t+{h}"
        real_col = f"real_t+{h}"
        ax.plot(df["date"], df[pred_col], label="预测", linewidth=1.4)
        ax.plot(df["date"], df[real_col], label="真实", linewidth=1.2)
        ax.set_title(f"T+{h} 预测值 vs 真实值")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")

    fig.add_subplot(gs[2, 1]).axis("off")

    hm_gs = gs[3, :].subgridspec(2, 2, wspace=0.25, hspace=0.35)
    hm_axes = [
        fig.add_subplot(hm_gs[0, 0]),
        fig.add_subplot(hm_gs[0, 1]),
        fig.add_subplot(hm_gs[1, 0]),
        fig.add_subplot(hm_gs[1, 1]),
    ]
    metric_order = ["price_corr", "ret_corr", "price_mae", "ret_mae"]
    titles = [
        "价格相关性 (price_corr)",
        "收益相关性 (ret_corr)",
        "价格误差 (price_mae)",
        "收益误差 (ret_mae)",
    ]
    cmaps = ["RdBu_r", "RdBu_r", "YlOrRd", "YlOrRd"]
    vmins = [-1, -1, None, None]
    vmaxs = [1, 1, None, None]

    for ax, metric, title, cmap, vmin, vmax in zip(
        hm_axes, metric_order, titles, cmaps, vmins, vmaxs
    ):
        data = metrics_df.loc[[metric]].values
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(metrics_df.columns)), metrics_df.columns)
        ax.set_yticks([0], [metric])
        ax.set_title(title)
        for j, col in enumerate(metrics_df.columns):
            val = metrics_df.loc[metric, col]
            text = "nan" if pd.isna(val) else f"{val:.4f}"
            ax.text(j, 0, text, ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(os.path.basename(csv_path), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main() -> None:
    for index_name in INDICES.keys():
        for rank in [1, 2, 3]:
            file_name = f"final_predictions_{index_name}_rank{rank}.csv"
            csv_path = os.path.join(DATA_DIR, file_name)
            if not os.path.exists(csv_path):
                continue
            _plot_file(csv_path)


if __name__ == "__main__":
    main()