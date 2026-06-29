# -*- coding: utf-8 -*-
"""
scripts.common.models

轻量模型接口层，借鉴 Qlib model.base 设计。
为规则模型和未来 ML 模型提供统一的 fit/predict 接口。
"""

from scripts.common.models.base import AlphaModel
from scripts.common.models.rule_model import SimpleRuleModel

__all__ = ["AlphaModel", "SimpleRuleModel"]

# LightGBMModel 延迟导出：当 lightgbm 未安装时不会在 import 时失败，
# 只有在实际实例化时才会抛出 ImportError。
try:
    from scripts.common.models.lightgbm_model import LightGBMModel
    __all__.append("LightGBMModel")
except ImportError:
    pass
