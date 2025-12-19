"""
生成GAN数据生成模块的PowerPoint架构图 - 顶会论文风格
参考CVPR/NeurIPS等顶会论文的高质量模型图设计
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE

# 精心设计的配色方案（参考顶会论文）
COLORS = {
    'data': (230, 245, 255),       # 数据 - 极浅蓝
    'encoder': (200, 230, 255),    # 特征提取 - 浅蓝
    'generator': (255, 230, 230),  # 生成器 - 浅粉红
    'discriminator': (230, 230, 255), # 判别器 - 浅紫
    'loss': (255, 250, 235),       # 损失 - 米色
    'output': (230, 255, 230),     # 输出 - 浅绿
    'bg': (250, 250, 252),         # 背景 - 极浅灰
    'text': (40, 40, 40),          # 文字 - 深灰
    'border': (120, 120, 120),     # 边框 - 灰
    'arrow': (90, 90, 90),         # 箭头 - 深灰
    'dash': (150, 150, 150)        # 虚线 - 浅灰
}

def add_rounded_box(slide, left, top, width, height, text, color_key='data', 
                    font_size=10, bold=True, border_color=None):
    """添加圆角矩形框（用于数据）"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*COLORS[color_key])
    shape.line.color.rgb = RGBColor(*(border_color or COLORS['border']))
    shape.line.width = Pt(1.5)
    
    text_frame = shape.text_frame
    text_frame.text = text
    text_frame.word_wrap = True
    text_frame.vertical_anchor = 1
    text_frame.margin_top = Inches(0.08)
    text_frame.margin_bottom = Inches(0.08)
    
    paragraph = text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    paragraph.font.size = Pt(font_size)
    paragraph.font.bold = bold
    paragraph.font.name = 'Arial'
    paragraph.font.color.rgb = RGBColor(*COLORS['text'])
    
    return shape

def add_rect_box(slide, left, top, width, height, text, color_key='encoder', 
                 font_size=10, bold=True):
    """添加矩形框（用于模块）"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*COLORS[color_key])
    shape.line.color.rgb = RGBColor(*COLORS['border'])
    shape.line.width = Pt(1.5)
    
    text_frame = shape.text_frame
    text_frame.text = text
    text_frame.word_wrap = True
    text_frame.vertical_anchor = 1
    text_frame.margin_top = Inches(0.08)
    text_frame.margin_bottom = Inches(0.08)
    
    paragraph = text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    paragraph.font.size = Pt(font_size)
    paragraph.font.bold = bold
    paragraph.font.name = 'Arial'
    paragraph.font.color.rgb = RGBColor(*COLORS['text'])
    
    return shape

def add_arrow(slide, x1, y1, x2, y2, label="", width=2.0, dashed=False, color=None):
    """添加箭头连接线"""
    connector = slide.shapes.add_connector(
        1,  # MSO_CONNECTOR.STRAIGHT
        Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    
    arrow_color = color or (COLORS['dash'] if dashed else COLORS['arrow'])
    connector.line.color.rgb = RGBColor(*arrow_color)
    connector.line.width = Pt(width)
    
    if dashed:
        connector.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    
    if label:
        label_w = max(0.6, len(label) * 0.065)
        label_h = 0.22
        label_x = (x1 + x2) / 2 - label_w / 2
        label_y = (y1 + y2) / 2 - label_h / 2
        
        bg_box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(label_x), Inches(label_y),
            Inches(label_w), Inches(label_h)
        )
        bg_box.fill.solid()
        bg_box.fill.fore_color.rgb = RGBColor(255, 255, 255)
        bg_box.line.color.rgb = RGBColor(*COLORS['border'])
        bg_box.line.width = Pt(0.75)
        
        text_frame = bg_box.text_frame
        text_frame.text = label
        text_frame.vertical_anchor = 1
        p = text_frame.paragraphs[0]
        p.font.size = Pt(8)
        p.font.name = 'Arial'
        p.font.color.rgb = RGBColor(*COLORS['text'])
        p.alignment = PP_ALIGN.CENTER

def create_gan_architecture_ppt():
    """创建高质量的GAN架构图"""
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # 添加背景色
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(*COLORS['bg'])
    
    # 主标题
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(12.3), Inches(0.5))
    title_frame = title_box.text_frame
    title_frame.text = "Cross-Domain GAN-based Data Augmentation Framework"
    title_p = title_frame.paragraphs[0]
    title_p.font.size = Pt(24)
    title_p.font.bold = True
    title_p.font.name = 'Arial'
    title_p.alignment = PP_ALIGN.CENTER
    title_p.font.color.rgb = RGBColor(25, 45, 110)
    
    # 副标题
    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.55), Inches(12.3), Inches(0.3))
    subtitle_frame = subtitle_box.text_frame
    subtitle_frame.text = "基于生成对抗网络的跨域CSI数据增强架构"
    subtitle_p = subtitle_frame.paragraphs[0]
    subtitle_p.font.size = Pt(13)
    subtitle_p.font.name = 'SimHei'
    subtitle_p.alignment = PP_ALIGN.CENTER
    subtitle_p.font.color.rgb = RGBColor(80, 80, 80)
    
    # ========== 第一列：输入数据 ==========
    col1_x = 0.6
    
    # 源域数据
    add_rounded_box(slide, col1_x, 1.3, 1.4, 0.65, 
                    "Source Domain\nx_s\n\nEnv0 + Env1", 
                    'data', font_size=9)
    
    # 目标域数据
    add_rounded_box(slide, col1_x, 2.2, 1.4, 0.65, 
                    "Target Domain\nx_t\n\nEnv2 (Few-shot)", 
                    'data', font_size=9)
    
    # ========== 第二列：特征提取器 ==========
    col2_x = 2.3
    
    feature_box = add_rect_box(slide, col2_x, 1.5, 1.35, 1.2, 
                                "Feature Extractor\nE\n\n1D-CNN\nLightweight\n\nExtract f_t\n(Env Fingerprint)", 
                                'encoder', font_size=9)
    
    # ========== 第三列：生成器（核心）==========
    col3_x = 4.0
    
    # 生成器主框
    gen_main = add_rect_box(slide, col3_x, 1.0, 2.1, 2.2, "", 'generator', font_size=9)
    
    # 生成器内部标题
    gen_title = slide.shapes.add_textbox(Inches(col3_x + 0.05), Inches(1.05), Inches(2.0), Inches(0.3))
    gen_title_frame = gen_title.text_frame
    gen_title_frame.text = "Generator G (U-Net)"
    gen_title_p = gen_title_frame.paragraphs[0]
    gen_title_p.font.size = Pt(11)
    gen_title_p.font.bold = True
    gen_title_p.font.name = 'Arial'
    gen_title_p.alignment = PP_ALIGN.CENTER
    gen_title_p.font.color.rgb = RGBColor(140, 40, 40)
    
    # 生成器内部结构
    gen_content = slide.shapes.add_textbox(Inches(col3_x + 0.15), Inches(1.45), Inches(1.8), Inches(1.6))
    gen_content_frame = gen_content.text_frame
    gen_content_frame.text = """Encoder
  ↓ Compress x_s
  ↓ Low-dim features

⊕ Fusion Layer
  Concat(enc, f_t)

Decoder
  ↑ Reconstruct
  ↑ U-Net skip connections
  
Output: x̃_t"""
    gen_content_p = gen_content_frame.paragraphs[0]
    gen_content_p.font.size = Pt(8.5)
    gen_content_p.font.name = 'Arial'
    gen_content_p.alignment = PP_ALIGN.LEFT
    gen_content_p.font.color.rgb = RGBColor(*COLORS['text'])
    gen_content_p.line_spacing = 1.1
    
    # ========== 第四列：合成样本 ==========
    col4_x = 6.4
    
    synthetic_box = add_rounded_box(slide, col4_x, 1.75, 1.3, 0.65, 
                                    "Synthetic\nx̃_t\n\nTarget-style\nFake CSI", 
                                    'output', font_size=9)
    
    # ========== 第五列：双判别器 ==========
    col5_x = 8.1
    
    # 时域判别器
    d_time = add_rect_box(slide, col5_x, 1.2, 1.5, 0.7, 
                          "Time-Domain\nDiscriminator D\n\nConv1D + SN", 
                          'discriminator', font_size=9)
    
    # 频域判别器
    d_freq = add_rect_box(slide, col5_x, 2.15, 1.5, 0.7, 
                          "Freq-Domain\nDiscriminator Ds\n\nFFT + Conv1D + SN", 
                          'discriminator', font_size=9)
    
    # 真实样本标注
    real_label = add_rounded_box(slide, col5_x, 3.1, 1.5, 0.4, 
                                 "Real x_t", 
                                 'data', font_size=8)
    
    # ========== 数据流箭头（实线）==========
    # x_s → Generator
    add_arrow(slide, 2.0, 1.62, 4.0, 1.5, "x_s", width=2.0)
    
    # x_t → Feature Extractor
    add_arrow(slide, 2.0, 2.52, 2.3, 2.3, "x_t", width=2.0)
    
    # Feature Extractor → Generator
    add_arrow(slide, 3.65, 2.1, 4.5, 2.3, "f_t", width=2.0)
    
    # Generator → Synthetic
    add_arrow(slide, 6.1, 2.1, 6.4, 2.08, "x̃_t", width=2.0)
    
    # Synthetic → D_time
    add_arrow(slide, 7.7, 1.95, 8.1, 1.5, "", width=2.0)
    
    # Synthetic → D_freq
    add_arrow(slide, 7.7, 2.2, 8.1, 2.5, "", width=2.0)
    
    # Real x_t → Discriminators (from bottom)
    add_arrow(slide, 8.85, 3.1, 8.85, 2.85, "x_t", width=2.0)
    
    # ========== 损失函数区域 ==========
    loss_y = 4.0
    
    # 损失函数背景框
    loss_bg = add_rect_box(slide, 0.6, loss_y, 9.0, 2.5, "", 'loss', font_size=8, bold=False)
    
    # 损失函数标题
    loss_title_box = slide.shapes.add_textbox(Inches(0.7), Inches(loss_y + 0.08), Inches(8.8), Inches(0.35))
    loss_title_frame = loss_title_box.text_frame
    loss_title_frame.text = "Loss Functions & Optimization Objectives"
    loss_title_p = loss_title_frame.paragraphs[0]
    loss_title_p.font.size = Pt(13)
    loss_title_p.font.bold = True
    loss_title_p.font.name = 'Arial'
    loss_title_p.alignment = PP_ALIGN.CENTER
    loss_title_p.font.color.rgb = RGBColor(120, 60, 20)
    
    # 损失函数内容（分两列）
    # 左列
    loss_left = slide.shapes.add_textbox(Inches(0.8), Inches(loss_y + 0.5), Inches(4.2), Inches(1.85))
    loss_left_frame = loss_left.text_frame
    loss_left_frame.text = """① Adversarial Loss (WGAN-GP)
   𝓛_adv = E[D(x_t)] - E[D(x̃_t)] + λ_GP·GP
   • Wasserstein distance with gradient penalty
   • Stabilize GAN training dynamics

② Frequency Consistency Loss
   𝓛_freq = ||FFT(x_t) - FFT(x̃_t)||₂²
   • Match spectral characteristics
   • Preserve frequency-domain patterns"""
    loss_left_p = loss_left_frame.paragraphs[0]
    loss_left_p.font.size = Pt(8)
    loss_left_p.font.name = 'Arial'
    loss_left_p.alignment = PP_ALIGN.LEFT
    loss_left_p.font.color.rgb = RGBColor(*COLORS['text'])
    loss_left_p.line_spacing = 1.15
    
    # 右列
    loss_right = slide.shapes.add_textbox(Inches(5.2), Inches(loss_y + 0.5), Inches(4.2), Inches(1.85))
    loss_right_frame = loss_right.text_frame
    loss_right_frame.text = """③ MMD Loss (Maximum Mean Discrepancy)
   𝓛_MMD = ||μ(f_t) - μ(f̃_t)||²_ℋ
   • Multi-scale Gaussian kernels
   • Align feature distributions

④ Content Preservation Loss
   𝓛_content = ||E(x_s) - E(x̃_t)||₁
   • Preserve gait identity from source domain
   • Maintain discriminative information"""
    loss_right_p = loss_right_frame.paragraphs[0]
    loss_right_p.font.size = Pt(8)
    loss_right_p.font.name = 'Arial'
    loss_right_p.alignment = PP_ALIGN.LEFT
    loss_right_p.font.color.rgb = RGBColor(*COLORS['text'])
    loss_right_p.line_spacing = 1.15
    
    # ========== 损失反馈箭头（虚线）==========
    # Discriminators → Generator (adversarial feedback)
    add_arrow(slide, 8.5, 1.2, 5.0, 1.0, "𝓛_adv", width=1.8, dashed=True, color=(200, 100, 100))
    add_arrow(slide, 8.5, 2.85, 5.5, 3.2, "𝓛_adv", width=1.8, dashed=True, color=(200, 100, 100))
    
    # Loss module → Generator
    add_arrow(slide, 4.8, 4.0, 5.0, 3.2, "𝓛_freq+𝓛_content", width=1.8, dashed=True, color=(100, 100, 200))
    
    # Loss module → Feature Extractor
    add_arrow(slide, 2.9, 4.0, 2.9, 2.7, "𝓛_MMD", width=1.8, dashed=True, color=(100, 150, 100))
    
    # ========== 图例和说明 ==========
    legend_x = 10.0
    
    # 图例框
    legend_bg = add_rect_box(slide, legend_x, 1.1, 2.7, 2.2, "", 'loss', font_size=8, bold=False)
    
    legend_box = slide.shapes.add_textbox(Inches(legend_x + 0.15), Inches(1.2), Inches(2.4), Inches(2.0))
    legend_frame = legend_box.text_frame
    legend_frame.text = """Visual Legend

━━━► Forward Data Flow
        (Solid arrows)

- - - ► Loss Gradient Feedback
        (Dashed arrows)

⬭  Rounded: Data/Samples

▭  Rectangle: Modules

Color Coding:
  Blue: Feature Extraction
  Pink: Generation
  Purple: Discrimination
  Green: Output"""
    legend_p = legend_frame.paragraphs[0]
    legend_p.font.size = Pt(8)
    legend_p.font.name = 'Arial'
    legend_p.alignment = PP_ALIGN.LEFT
    legend_p.font.color.rgb = RGBColor(*COLORS['text'])
    legend_p.line_spacing = 1.2
    
    # 关键设计说明
    design_bg = add_rect_box(slide, legend_x, 3.6, 2.7, 2.9, "", 'loss', font_size=8, bold=False)
    
    design_box = slide.shapes.add_textbox(Inches(legend_x + 0.15), Inches(3.7), Inches(2.4), Inches(2.65))
    design_frame = design_box.text_frame
    design_frame.text = """Key Design Features

✓ U-Net Architecture
  Skip connections preserve
  fine-grained details

✓ Dual Discriminators
  Time & Frequency domain
  comprehensive constraints

✓ Feature Extractor
  Lightweight 1D-CNN for
  environment adaptation

✓ Multi-objective Learning
  4-fold loss for robust
  domain adaptation

✓ WGAN-GP Framework
  Stable adversarial training"""
    design_p = design_frame.paragraphs[0]
    design_p.font.size = Pt(7.5)
    design_p.font.name = 'Arial'
    design_p.alignment = PP_ALIGN.LEFT
    design_p.font.color.rgb = RGBColor(*COLORS['text'])
    design_p.line_spacing = 1.15
    
    # 保存文件
    output_path = 'Research1/GAN数据生成模块架构图_优化版.pptx'
    prs.save(output_path)
    print(f"✓ 已生成优化版架构图: {output_path}")
    print("✓ 采用顶会论文级别的视觉设计")
    print("✓ 清晰的模块划分和数据流向")
    print("✓ 专业的配色方案和排版布局")
    return output_path

if __name__ == "__main__":
    create_gan_architecture_ppt()
