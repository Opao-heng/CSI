"""
TFGAN-CAL 消融实验模块

包含5个模型变体的实现和评估:
- Model A: Baseline (w/o TFGAN & CAL)
- Model B: w/o Generator, Only CAL
- Model C: w/o Freq-D, Time-only TFGAN
- Model D: w/o AdaIN, Concat-only TFGAN
- Model E: TFGAN-CAL (Full)
"""

__version__ = '1.0.0'
__author__ = 'TFGAN-CAL Research Team'
