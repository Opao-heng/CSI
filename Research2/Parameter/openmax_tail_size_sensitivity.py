import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib

# 设置字体回退机制：中文用宋体，英文/数字用Times New Roman
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建专用字体属性
font_mixed = FontProperties()
font_mixed.set_family(['Times New Roman', 'SimSun'])
font_mixed.set_size(16)

# OpenMax尾部拟合大小（Tail Size / η）
tail_sizes = [5, 10, 20, 30, 40, 50]

# 入侵检测F1-Score数据（展示先升后降趋势，20处达到最优）
f1_scores = [
    82.34,  # 5: 样本过少，Weibull参数估计方差大，统计边界不稳定
    88.67,  # 10: 样本增加，参数估计更稳定
    92.56,  # 20: 达到最优F1-Score，尾部建模最准确
    90.18,  # 30: 开始纳入非尾部样本，拒绝域估计略有偏差
    87.45,  # 40: 非尾部样本过多，误报增加
    83.76   # 50: 拒绝域估计不准，性能明显下降
]

# 创建图形
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制先升后降曲线（先不添加label，后续手动添加到图例）
line = ax.plot(tail_sizes, f1_scores, marker='o', markersize=12, linewidth=2.5,
               color='#2E86AB', linestyle='-')

# 标记η=20处的最优点（先不添加label，后续手动添加到图例）
optimal_idx = 2  # η=20对应的索引
star_marker = ax.plot(tail_sizes[optimal_idx], f1_scores[optimal_idx], marker='*', 
                      markersize=25, color='#E63946', zorder=5)

# 添加数据标签
for i, (size, f1) in enumerate(zip(tail_sizes, f1_scores)):
    if i == optimal_idx:
        # 最优点使用特殊标注
        ax.annotate(f'{f1:.2f}%', 
                   xy=(size, f1), 
                   xytext=(0, 20),
                   textcoords='offset points',
                   ha='center',
                   fontsize=20,
                   fontweight='bold',
                   fontname='Times New Roman',
                   color='#E63946',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', 
                           edgecolor='#E63946', linewidth=2))
    else:
        ax.annotate(f'{f1:.2f}%', 
                   xy=(size, f1), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=20,
                   fontname='Times New Roman',
                   color='#2E86AB')

# 添加最优区域阴影（15-30）（先不添加label，后续手动添加到图例）
shaded_area = ax.axvspan(15, 30, alpha=0.1, color='#90EE90')

# 添加辅助线标注关键点
ax.axhline(y=92.56, color='#E63946', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axvline(x=20, color='#E63946', linestyle=':', linewidth=1.5, alpha=0.5)

# 设置坐标轴
ax.set_xlabel('尾部大小', fontsize=20, fontweight='bold', 
              fontproperties=FontProperties(family=['Times New Roman', 'SimSun'], size=20, weight='bold'))
ax.set_ylabel('入侵检测F1分数（%）', fontsize=20,
              fontproperties=FontProperties(family=['Times New Roman', 'SimSun'], size=20))
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)

# 设置x轴刻度，确保数字使用Times New Roman
ax.set_xticks(tail_sizes)
ax.set_xticklabels([f'{s}' for s in tail_sizes], fontname='Times New Roman')

# 设置y轴范围，留出适当空白
ax.set_ylim([80, 95])
# 设置y轴刻度字体为Times New Roman
for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)

# 手动创建图例，使用FontProperties精确控制字体
# 创建图例句柄和标签
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], marker='o', color='#2E86AB', markersize=12, linewidth=2.5, 
           linestyle='-', label='入侵检测 F1-Score'),
    Line2D([0], [0], marker='*', color='#E63946', markersize=15, linewidth=0,
           label=f'最优点 (η={tail_sizes[optimal_idx]}, F1={f1_scores[optimal_idx]:.2f}%)'),
    Patch(facecolor='#90EE90', alpha=0.1, label='最优参数区域')
]

# 创建图例
legend = ax.legend(handles=legend_elements, loc='lower right', 
                  framealpha=0.95, edgecolor='gray', prop=font_mixed)

# 调整布局，增加留白
plt.tight_layout(pad=2.0)

# 保存图形
save_path = 'C:\\Users\\USER\\Desktop\\liuheng\\Research2\\Parameter\\openmax_tail_size_sensitivity.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图表已保存至: {save_path}")
