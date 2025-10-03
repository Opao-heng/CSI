"""
入侵者检测模块
实现基于WiFi CSI数据的身份认证和入侵者检测系统
"""

from .model import IntruderDetectionSystem, FeatureExtractor, ContrastiveDomainAdapter, IdentityClassifier
from .loss import IntruderDetectionLoss
from .data_loader import load_and_split_data, create_data_loaders

__all__ = [
    'IntruderDetectionSystem',
    'FeatureExtractor',
    'ContrastiveDomainAdapter', 
    'IdentityClassifier',
    'IntruderDetectionLoss',
    'load_and_split_data',
    'create_data_loaders'
]