# 如何新增自訂 Prompt

## 快速開始

如果你想新增不同類型的報告（例如：簡要摘要、家長指引等），請按照以下步驟：

## 步驟 1: 創建新的 Prompt 檔案

在 `prompts/` 目錄下創建新的 Python 檔案，例如 `brief_summary.py`：

```python
"""
簡要摘要報告生成 Prompt

此模組提供簡化版本的職能治療報告，適合快速瀏覽。
"""

def get_system_prompt():
    """
    取得系統 prompt
    
    Returns:
        str: 系統 prompt 文字
    """
    return """你是一位專業的職能治療師 (OT)。
你的任務是根據個案資料，生成一份「簡要摘要」報告。

規則：
1. 必須使用台灣繁體中文
2. 內容精簡，每個領域不超過 2 句話
3. 重點突出核心問題
"""


def get_user_prompt(context_str, case_description):
    """
    取得使用者 prompt
    
    Args:
        context_str (str): 參考案例
        case_description (str): 個案描述
    
    Returns:
        str: 使用者 prompt 文字
    """
    return f"""
參考案例：
{context_str}

個案資料：
{case_description}

請生成簡要摘要（每個領域 1-2 句話）：
1. 核心問題
2. 建議方向
"""


def get_prompt_metadata():
    """取得 prompt 元資料"""
    return {
        "version": "1.0",
        "name": "簡要摘要報告",
        "description": "精簡版問題分析",
        "created_date": "2026-02-04",
        "output_sections": ["核心問題", "建議方向"]
    }
```

## 步驟 2: 更新 `__init__.py`

編輯 `prompts/__init__.py`，加入新的 prompt：

```python
from .standard_report import (
    get_system_prompt as get_standard_system_prompt,
    get_user_prompt as get_standard_user_prompt,
)

from .brief_summary import (
    get_system_prompt as get_brief_system_prompt,
    get_user_prompt as get_brief_user_prompt,
)

__all__ = [
    'get_standard_system_prompt',
    'get_standard_user_prompt',
    'get_brief_system_prompt',
    'get_brief_user_prompt',
]
```

## 步驟 3: 在 app.py 中使用

修改 `app.py` 來使用新的 prompt：

```python
# 導入新的 prompt
from prompts import (
    get_standard_system_prompt,
    get_standard_user_prompt,
    get_brief_system_prompt,
    get_brief_user_prompt
)

# 在生成函式中選擇使用哪個 prompt
def generate_report(case_description, report_type="standard"):
    # ... 檢索邏輯 ...
    
    # 根據類型選擇 prompt
    if report_type == "brief":
        system_prompt = get_brief_system_prompt()
        user_prompt = get_brief_user_prompt(context_str, case_description)
    else:
        system_prompt = get_standard_system_prompt()
        user_prompt = get_standard_user_prompt(context_str, case_description)
    
    # ... 生成邏輯 ...
```

## 步驟 4: 測試新 Prompt

1. 確保新的 prompt 檔案語法正確
2. 測試生成結果是否符合預期
3. 在 `examples/` 目錄下新增範例輸出

## Prompt 設計最佳實踐

### ✅ 應該做的事

1. **保持函式介面一致**：所有 prompt 模組都應該有 `get_system_prompt()` 和 `get_user_prompt()`
2. **加入詳細的 docstring**：說明每個函式的用途和參數
3. **提供元資料**：使用 `get_prompt_metadata()` 記錄版本和描述
4. **遵循專業規範**：確保使用台灣職能治療專業術語
5. **測試輸出品質**：在 `examples/` 中記錄實際輸出範例

### ❌ 應該避免的事

1. **不要硬編碼在 app.py**：所有 prompt 都應該在獨立檔案中
2. **不要混用簡體中文**：嚴格使用台灣繁體中文
3. **不要過度複雜**：每個 prompt 模組應該專注於單一類型的報告
4. **不要忽略錯誤處理**：確保 prompt 能處理邊界情況

## 目錄結構

```
.agent/skills/ot-report-generation/
├── SKILL.md                    # Skill 主要說明
├── README.md                   # 本檔案
├── prompts/
│   ├── __init__.py            # 模組初始化
│   ├── standard_report.py     # 標準報告
│   ├── brief_summary.py       # 簡要摘要（範例）
│   └── parent_guide.py        # 家長指引（範例）
└── examples/
    └── sample_outputs.md      # 範例輸出
```

## 常見問題

### Q: 如何修改現有的 prompt？

直接編輯對應的 `.py` 檔案，修改 `get_system_prompt()` 或 `get_user_prompt()` 的內容。

### Q: 如何在不同 prompt 之間切換？

在 `app.py` 中加入選擇邏輯，或在 Gradio 介面中加入下拉選單讓使用者選擇。

### Q: 如何確保 prompt 品質？

1. 參考 `examples/sample_outputs.md` 中的範例
2. 測試多個不同的個案描述
3. 請專業職能治療師審核輸出內容

## 版本控制建議

當你修改 prompt 時，建議：

1. 更新 `get_prompt_metadata()` 中的版本號
2. 在 Git commit 中說明修改原因
3. 保留舊版本的輸出範例以便比較

## 需要協助？

如果在新增 prompt 時遇到問題，可以：

1. 參考 `standard_report.py` 的實作
2. 查看 `SKILL.md` 了解整體架構
3. 檢查 `examples/sample_outputs.md` 確認輸出格式
