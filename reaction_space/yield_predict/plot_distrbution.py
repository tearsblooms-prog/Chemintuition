import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 读取数据
# ======================
df = pd.read_csv("yield_distribution.csv")

# ======================
# 全局绘图配置
# ======================
plt.rcParams.update({
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
    "font.weight": "bold"
})

# ======================
# 配色
# ======================
custom_colors = ["#00b894", "#ff7675", "#fdcb6e", "#fd79a8", "#636e72",'#E3F2D9']

# ======================
# 绘制柱状图
# ======================
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(
    df["yield_group"],
    df["count"],
    color=custom_colors[:len(df)]
)

# ======================
# 去除坐标轴与说明
# ======================
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# ======================
# 柱子顶部数值标注
# ======================
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=24,
        fontweight="bold"
    )

# ======================
# 保存为 SVG（透明背景）
# ======================
plt.tight_layout()
plt.savefig("yield_distribution.svg", format="svg", transparent=True)
plt.close()
