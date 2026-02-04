# OT Report Generation Skill - 架構總覽

## 📁 目錄結構

```
.agent/skills/ot-report-generation/
│
├── 📄 SKILL.md                          # Skill 主要說明文件
├── 📄 README.md                         # 如何新增自訂 Prompt 的指南
│
├── 📂 prompts/                          # Prompt 模組目錄
│   ├── __init__.py                      # 模組初始化（方便導入）
│   └── standard_report.py               # 標準報告 Prompt
│       ├── get_system_prompt()          # 系統角色定義
│       ├── get_user_prompt()            # 使用者查詢模板
│       └── get_prompt_metadata()        # Prompt 元資料
│
└── 📂 examples/                         # 範例輸出
    └── sample_outputs.md                # 標準報告範例
```

## 🔄 工作流程

```
使用者輸入個案描述
        ↓
app.py 進行分區檢索 (Decomposed RAG)
        ↓
從向量資料庫取得相關案例 (context_str)
        ↓
載入 Skill Prompt 模組
        ↓
┌─────────────────────────────────────┐
│ from prompts import                 │
│   get_system_prompt,                │
│   get_user_prompt                   │
└─────────────────────────────────────┘
        ↓
組合 Prompt
        ↓
┌─────────────────────────────────────┐
│ system_prompt = get_system_prompt() │
│ user_prompt = get_user_prompt(      │
│     context_str,                    │
│     case_description                │
│ )                                   │
└─────────────────────────────────────┘
        ↓
呼叫 LLM (Ollama qwen2.5:7b)
        ↓
串流輸出報告
```

## ✨ 核心優勢

### 1. **模組化管理**
- ✅ Prompt 邏輯獨立於應用程式碼
- ✅ 易於版本控制和追蹤變更
- ✅ 可在不同專案間重用

### 2. **可擴展性**
- ✅ 新增報告類型只需新增一個 `.py` 檔案
- ✅ 不需要修改核心應用邏輯
- ✅ 支援多種報告格式並存

### 3. **可維護性**
- ✅ 集中管理所有 Prompt
- ✅ 清晰的函式介面
- ✅ 完整的文檔和範例

### 4. **專業性**
- ✅ 嚴格遵循台灣職能治療專業術語
- ✅ 臨床推理邏輯一致性
- ✅ 輸出格式標準化

## 🚀 如何使用

### 在 app.py 中使用（已整合）

```python
import sys
import os

# 設定 Skill 路徑
SKILL_PATH = os.path.join(os.path.dirname(__file__), 
                          '.agent', 'skills', 'ot-report-generation')
sys.path.insert(0, SKILL_PATH)

# 導入 Prompt
from prompts import get_system_prompt, get_user_prompt

# 使用 Prompt
system_prompt = get_system_prompt()
user_prompt = get_user_prompt(context_str, case_description)
```

### 新增自訂報告類型

1. 在 `prompts/` 下創建新檔案（如 `brief_summary.py`）
2. 實作 `get_system_prompt()` 和 `get_user_prompt()`
3. 更新 `prompts/__init__.py` 加入新的導入
4. 在 `app.py` 中選擇使用哪個 Prompt

詳細步驟請參考 `README.md`

## 📊 目前支援的報告類型

| 類型 | 檔案 | 說明 | 狀態 |
|------|------|------|------|
| 標準報告 | `standard_report.py` | 完整的問題分析與治療建議 | ✅ 已實作 |
| 簡要摘要 | `brief_summary.py` | 精簡版快速瀏覽 | 📝 待新增 |
| 家長指引 | `parent_guide.py` | 家長友善版本 | 📝 待新增 |
| 進度追蹤 | `progress_report.py` | 療程進度報告 | 📝 待新增 |

## 🔧 維護指南

### 修改現有 Prompt
1. 編輯對應的 `.py` 檔案
2. 更新 `get_prompt_metadata()` 中的版本號
3. 測試輸出品質
4. 更新 `examples/` 中的範例

### 版本控制
- 每次重大修改都應更新版本號
- 在 Git commit 中詳細說明變更原因
- 保留舊版本的輸出範例以便比較

## 📝 設計原則

1. **一致性**：所有 Prompt 模組遵循相同的函式介面
2. **專業性**：確保使用正確的台灣職能治療術語
3. **可測試性**：每個 Prompt 都應有對應的範例輸出
4. **文檔化**：詳細的 docstring 和使用說明

## 🎯 未來規劃

- [ ] 新增簡要摘要報告
- [ ] 新增家長指引報告
- [ ] 新增進度追蹤報告
- [ ] 實作 Prompt 版本切換功能
- [ ] 建立 Prompt 效能評估機制
- [ ] 支援多語言輸出（如英文版）

---

**版本**: 1.0  
**建立日期**: 2026-02-04  
**最後更新**: 2026-02-04
