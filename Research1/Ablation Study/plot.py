import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ===========================
# 字体配置：中文宋体 + 英文Times New Roman
# ===========================
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'

# 设置全局字体大小为20
FONTSIZE = 20
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['axes.labelsize'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams['legend.fontsize'] = FONTSIZE


def plot_ablation_study():
    """
    绘制消融实验结果的簇状柱状图
    图名：各核心模块对模型性能的贡献度分析
    """
    # ===========================
    # 数据定义（根据表3-3）
    models = ['Model F\n完整模型', 'Model E', 'Model C',  'Model B', 'Model D', 'Model A']
    
    # 源域识别准确率
    source_accuracy = [97.36, 93.34, 92.89, 90.85, 94.82, 88.79]
    
    # 跨域识别准确率
    cross_accuracy = [99.00, 54.89, 72.78, 68.90, 64.39, 79.24]
    
    # FID分数（越低越好）
    fid_scores = [89.72, np.nan, 105.34, 128.45, np.nan, 114.89]
    
    # IS分数（越高越好）
    is_scores = [7.83, np.nan, 5.62, 6.15, np.nan, 6.89]
    
    # ===========================
    # 绘图配置
    # ===========================
    x = np.arange(len(models))  # 模型标签位置
    width = 0.2  # 每个柱子的宽度
    
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 学术配色方案（色盲友好）
    colors = {
        'source': '#E64B35',     # 红色 - 源域识别
        'cross': '#4DBBD5',      # 蓝色 - 跨域识别
        'fid': '#00A087',        # 绿色 - FID
        'is': '#3C5488'          # 深蓝 - IS
    }
    
    # ===========================
    # 绘制簇状柱状图
    # ===========================
    bars1 = ax.bar(x - 1.5*width, source_accuracy, width, label='源域识别准确率(%)', 
                   color=colors['source'], edgecolor='black', linewidth=1.2, alpha=0.85)
    
    bars2 = ax.bar(x - 0.5*width, cross_accuracy, width, label='跨域识别准确率(%)', 
                   color=colors['cross'], edgecolor='black', linewidth=1.2, alpha=0.85)
    
    bars3 = ax.bar(x + 0.5*width, fid_scores, width, label='FID(↓)', 
                   color=colors['fid'], edgecolor='black', linewidth=1.2, alpha=0.85)
    
    bars4 = ax.bar(x + 1.5*width, is_scores, width, label='IS(↑)', 
                   color=colors['is'], edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # ===========================
    # 添加数据标签
    # ===========================
    def add_value_labels(bars, values):
        """在柱子顶部添加数值标签"""
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=14, 
                       fontweight='bold', color='black')
    
    add_value_labels(bars1, source_accuracy)
    add_value_labels(bars2, cross_accuracy)
    add_value_labels(bars3, fid_scores)
    add_value_labels(bars4, is_scores)
    
    # ===========================
    # 图表美化
    # ===========================
    # 设置坐标轴标签
    # ax.set_xlabel('模型变体', fontsize=FONTSIZE, fontweight='bold',
    #               fontproperties=font_manager.FontProperties(family='SimSun', size=FONTSIZE))
    ax.set_ylabel('指标数值', fontsize=FONTSIZE, fontweight='bold',
                  fontproperties=font_manager.FontProperties(family='SimSun', size=FONTSIZE))
    
    # 设置图标题
    ax.set_title('各核心模块对模型性能的贡献度分析', fontsize=FONTSIZE+2, fontweight='bold', pad=20,
                fontproperties=font_manager.FontProperties(family='SimSun', size=FONTSIZE+2))
    
    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=16, 
                       fontproperties=font_manager.FontProperties(family='SimSun', size=16))
    
    # 设置y轴范围和刻度
    ax.set_ylim(0, 140)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=1)
    ax.set_axisbelow(True)
    
    # 添加图例
    legend = ax.legend(loc='upper right', frameon=True, shadow=True, 
                      fancybox=True, framealpha=0.9, fontsize=FONTSIZE-2,
                      prop=font_manager.FontProperties(family='SimSun', size=FONTSIZE-2))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    
    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # ===========================
    # 添加关键标注
    # ===========================
    # 标注完整模型（Model F）
    ax.annotate('完整模型\n性能最优', 
                xy=(0, 99), 
                xytext=(0.5, 115),
                fontsize=16,
                fontweight='bold',
                color='#E64B35',
                ha='center',
                fontproperties=font_manager.FontProperties(family='SimSun', size=16),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3, edgecolor='red', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, connectionstyle='arc3,rad=0.3'))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = 'c:/Users/USER/Desktop/liuheng/Research1/Ablation Study/ablation_study_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图表已保存至: {save_path}")
    
    # 显示图像
    # plt.show()


if __name__ == "__main__":
    plot_ablation_study()
