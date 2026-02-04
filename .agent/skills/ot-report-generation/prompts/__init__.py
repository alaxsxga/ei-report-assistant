"""
OT Report Generation Prompts Module

此模組包含各種職能治療報告生成的 prompt。

可用的 prompt 類型：
- standard_report: 標準完整報告（問題分析 + 治療建議）
"""

from .standard_report import (
    get_system_prompt,
    get_user_prompt,
    get_prompt_metadata
)

__all__ = [
    'get_system_prompt',
    'get_user_prompt',
    'get_prompt_metadata'
]

__version__ = '1.0.0'
