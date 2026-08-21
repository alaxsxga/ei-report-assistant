"""
OT Report Generation Prompts Module

此模組包含結構化報告生成所需的 prompt：
- 區塊拆解（segmentation）：把使用者輸入拆成領域區塊
- 結構化生成（json）：針對每個領域各自生成問題分析與建議
"""

from .standard_report import (
    get_json_system_prompt,
    get_json_user_prompt,
    get_segmentation_system_prompt,
    get_segmentation_user_prompt,
    get_prompt_metadata
)

__all__ = [
    'get_json_system_prompt',
    'get_json_user_prompt',
    'get_segmentation_system_prompt',
    'get_segmentation_user_prompt',
    'get_prompt_metadata'
]

__version__ = '1.0.0'
