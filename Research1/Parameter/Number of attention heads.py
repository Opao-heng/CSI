import matplotlib.pyplot as plt

# 设置中文字体和负号显示
# 中文采用宋体，英文/数字采用Times New Roman
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 注意力头数
num_heads = [2, 4, 6, 8, 10]

# 跨域准确率
accuracy_target = [
    91.52,  # H=2: 头数过少，特征提取能力不足
    93.33,  # H=4: 头数增加，性能提升
    97.36,  # H=6: 最优点，平衡了表达能力与计算效率
    95.43,  # H=8: 头数过多，开始过拟合
    94.89   # H=10: 头数过多，性能下降明显
]

# 源域准确率（通常高于跨域准确率）
accuracy_source = [
    89.21,  # H=2
    95.52,  # H=4
    99.00,  # H=6: 最优点
    93.63,  # H=8
    92.95   # H=10
]

# 创建图形
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制跨域准确率曲线
line1 = ax.plot(num_heads, accuracy_target, marker='o', markersize=10, linewidth=2.5, 
                color='#2E86AB', label='跨域准确率', linestyle='-')

# 绘制源域准确率曲线
line2 = ax.plot(num_heads, accuracy_source, marker='s', markersize=10, linewidth=2.5, 
                color='#A23B72', label='源域准确率', linestyle='--')

# 标记最优点（H=6）
optimal_idx = 2  # H=6对应的索引
ax.plot(num_heads[optimal_idx], accuracy_target[optimal_idx], marker='*', 
        markersize=20, color='#E63946', zorder=5, label=f'跨域最优 (H=6, Acc={accuracy_target[optimal_idx]}%)')
ax.plot(num_heads[optimal_idx], accuracy_source[optimal_idx], marker='*', 
        markersize=20, color='#F77F00', zorder=5, label=f'源域最优 (H=6, Acc={accuracy_source[optimal_idx]}%)')

# 定义字体族
font_zh = {'fontname': 'SimSun', 'fontsize': 20}  # 中文宋体
font_en = {'fontname': 'Times New Roman', 'fontsize': 20}  # 英文Times New Roman

# 添加跨域准确率数据标签
for i, (h, acc) in enumerate(zip(num_heads, accuracy_target)):
    if i == optimal_idx:
        ax.annotate(f'{acc:.2f}%', 
                   xy=(h, acc), 
                   xytext=(-15, 15),
                   textcoords='offset points',
                   ha='center',
                   fontsize=16,
                   fontweight='bold',
                   fontname='Times New Roman',
                   color='#E63946',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE5E5', edgecolor='#E63946', linewidth=1.5))
    else:
        ax.annotate(f'{acc:.2f}%', 
                   xy=(h, acc), 
                   xytext=(-10, 8),
                   textcoords='offset points',
                   ha='center',
                   fontsize=16,
                   fontname='Times New Roman',
                   color='#2E86AB')

# 添加源域准确率数据标签
for i, (h, acc) in enumerate(zip(num_heads, accuracy_source)):
    if i == optimal_idx:
        ax.annotate(f'{acc:.2f}%', 
                   xy=(h, acc), 
                   xytext=(15, 15),
                   textcoords='offset points',
                   ha='center',
                   fontsize=16,
                   fontweight='bold',
                   fontname='Times New Roman',
                   color='#F77F00',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE8CC', edgecolor='#F77F00', linewidth=1.5))
    else:
        ax.annotate(f'{acc:.2f}%', 
                   xy=(h, acc), 
                   xytext=(10, 8),
                   textcoords='offset points',
                   ha='center',
                   fontsize=16,
                   fontname='Times New Roman',
                   color='#A23B72')

# 添加最优区域阴影
ax.axvspan(5, 7, alpha=0.15, color='#90EE90', label='最优区域')

# 设置坐标轴
ax.set_xlabel('注意力头数', fontsize=20, fontweight='bold', fontname='SimSun')
ax.set_ylabel('识别准确率 (Accuracy) [%]', fontsize=20, fontweight='bold', fontname='SimSun')
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)

# 设置x轴刻度
ax.set_xticks(num_heads)
ax.set_xticklabels([f'H={h}' for h in num_heads])

# 设置y轴范围，留出适当空白
ax.set_ylim([88, 100])

# 添加网格线（更加细腻）
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)

# 添加图例 - 放在中下部位置，不遮挡数据
from matplotlib import font_manager as fm
legend = ax.legend(loc='lower center', fontsize=16, framealpha=0.95, edgecolor='gray', bbox_to_anchor=(0.5, 0.05))
for text in legend.get_texts():
    if any('\u4e00' <= c <= '\u9fff' for c in text.get_text()):
        text.set_fontname('SimSun')
        text.set_fontsize(16)
    else:
        text.set_fontname('Times New Roman')
        text.set_fontsize(16)


# 调整布局，增加留白
plt.tight_layout(pad=2.0)

# 保存图形
save_path = '/Research1/Parameter/attention_heads_sensitivity.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图表已保存至: {save_path}")


# 打印分析摘要
print("\n=== 注意力头数敏感性分析结果 ===")
print(f"测试头数范围: {min(num_heads)} - {max(num_heads)}")
print(f"最优头数: H={num_heads[optimal_idx]}")
print(f"跨域最优准确率: {accuracy_target[optimal_idx]}%")
print(f"源域最优准确率: {accuracy_source[optimal_idx]}%")
print(f"\n跨域准确率趋势分析:")
print(f"  - H=2到H=6: 准确率从{accuracy_target[0]}%提升至{accuracy_target[optimal_idx]}%（提升{accuracy_target[optimal_idx]-accuracy_target[0]:.1f}%）")
print(f"  - H=6到H=10: 准确率从{accuracy_target[optimal_idx]}%下降至{accuracy_target[-1]}%（下降{accuracy_target[optimal_idx]-accuracy_target[-1]:.1f}%）")
print(f"\n源域准确率趋势分析:")
print(f"  - H=2到H=6: 准确率从{accuracy_source[0]}%提升至{accuracy_source[optimal_idx]}%（提升{accuracy_source[optimal_idx]-accuracy_source[0]:.1f}%）")
print(f"  - H=6到H=10: 准确率从{accuracy_source[optimal_idx]}%下降至{accuracy_source[-1]}%（下降{accuracy_source[optimal_idx]-accuracy_source[-1]:.1f}%）")
print(f"\n结论: H=6在两个域中均达到最优，平衡了表达能力与计算效率")
