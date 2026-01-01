import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import matplotlib

# ===========================
# 字体配置：中文宋体 + 英文Times New Roman
# ===========================
plt.rcParams['axes.unicode_minus'] = False

# 先设置英文数字为Times New Roman
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

# 为中文文本准备宋体字体属性
CHINESE_FONT = font_manager.FontProperties(family='SimSun', size=20)

# 设置全局字体大小为20
FONTSIZE = 20
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['axes.labelsize'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
plt.rcParams['legend.fontsize'] = FONTSIZE

print("字体配置完成: Times New Roman(英文数字) + SimSun(中文)")


def plot_ablation_study():
    """
    绘制消融实验结果的簇状柱状图
    图名：各核心模块对模型性能的贡献度分析
    """
    # ===========================
    # 数据定义（根据表3-3）
    models = ['Model F', 'Model E', 'Model C', 'Model B', 'Model D', 'Model A']
    
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
    width = 0.18  # 增大柱子宽度，避免数据标签重叠
    
    fig, ax = plt.subplots(figsize=(20, 10))  # 增大图像宽度
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
    bars1 = ax.bar(x - 1.5*width, source_accuracy, width, label='源域识别准确率 (%)', 
                   color=colors['source'], edgecolor='#1a1a1a', linewidth=0.8, alpha=0.9)
    
    bars2 = ax.bar(x - 0.5*width, cross_accuracy, width, label='跨域识别准确率 (%)', 
                   color=colors['cross'], edgecolor='#1a1a1a', linewidth=0.8, alpha=0.9)
    
    bars3 = ax.bar(x + 0.5*width, fid_scores, width, label='FID (↓)', 
                   color=colors['fid'], edgecolor='#1a1a1a', linewidth=0.8, alpha=0.9)
    
    bars4 = ax.bar(x + 1.5*width, is_scores, width, label='IS (↑)', 
                   color=colors['is'], edgecolor='#1a1a1a', linewidth=0.8, alpha=0.9)
    
    # ===========================
    # 添加数据标签
    # ===========================
    def add_value_labels(bars, values):
        """在柱子顶部添加数值标签"""
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=16,
                       fontweight='normal', color='black')
    
    add_value_labels(bars1, source_accuracy)
    add_value_labels(bars2, cross_accuracy)
    add_value_labels(bars3, fid_scores)
    add_value_labels(bars4, is_scores)
    
    # ===========================
    # 图表美化
    # ===========================
    # 设置坐标轴标签（中文显式指定宋体）
    ax.set_ylabel('指标数值', fontproperties=CHINESE_FONT, labelpad=10)
    
    # 设置图标题（中文显式指定宋体）
    # ax.set_title('各核心模块对模型性能的贡献度分析', fontproperties=CHINESE_FONT, pad=20)
    
    # 设置x轴刻度（模型名称：英文用Times New Roman，中文用宋体）
    ax.set_xticks(x)
    # 分离处理：为英文和中文设置不同字体
    labels = []
    for model in models:
        if '完整模型' in model:
            # Model F 用Times New Roman，完整模型用宋体
            labels.append('Model F\n完整模型')
        else:
            labels.append(model)
    
    ax.set_xticklabels(labels)
    
    # 手动设置每个标签的字体
    for i, tick in enumerate(ax.get_xticklabels()):
        label_text = tick.get_text()
        if '完整模型' in label_text:
            # 包含中文的标签，需要使用宋体渲染中文部分
            tick.set_fontproperties(CHINESE_FONT)
        else:
            # 纯英文标签使用Times New Roman
            tick.set_fontsize(FONTSIZE)
            tick.set_fontname('Times New Roman')
    
    # 设置y轴范围和刻度
    ax.set_ylim(0, 140)
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.6, color='#cccccc')
    ax.set_axisbelow(True)
    
    # 设置y轴刻度样式（数字用Times New Roman）
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=6, width=1.0)
    
    # 添加图例（调整到左上角，避免遮挡数据）
    # 图例中含有中文，需要显式指定宋体
    legend = ax.legend(loc='upper left', frameon=True, shadow=False, 
                      fancybox=False, framealpha=1.0, 
                      prop=CHINESE_FONT,  # 使用宋体
                      edgecolor='#1a1a1a')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.0)
    
    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('#1a1a1a')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('#1a1a1a')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    save_path = 'c:/Users/USER/Desktop/liuheng/Research1/Ablation Study/ablation_study_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图表已保存至: {save_path}")



if __name__ == "__main__":
    plot_ablation_study()
