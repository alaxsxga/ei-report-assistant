---
name: Early Intervention Report Generation
description: 職能治療的早療報告生成系統 - 根據個案的評估，生成包含問題分析與治療建議的完整報告
---

# 職能治療的早療報告生成 Skill

## 概述

這個 skill 提供了職能治療的早療報告的自動生成能力，基於 RAG (Retrieval-Augmented Generation) 架構，結合歷史案例資料庫來產生專業的評估報告。

## 核心功能

1. **標準報告生成**：根據個案的評估，生成包含問題分析與治療建議的完整報告
2. **分區檢索 (Decomposed RAG)**：自動識別評估區塊，針對性檢索相關歷史案例
3. **專業術語一致性**：確保使用台灣職能治療專業術語

## 使用方式

### 在 Python 程式中使用

```python
from prompts.standard_report import get_system_prompt, get_user_prompt

# 取得 system prompt
system_prompt = get_system_prompt()

# 取得 user prompt（需要提供 context 和 case_description）
user_prompt = get_user_prompt(
    context_str="參考案例內容...",
    case_description="個案描述..."
)

# 使用 prompt 呼叫 LLM
response = llm.chat(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)
```

## Prompt 模組說明

### `prompts/standard_report.py`

標準報告生成的 prompt 模組，包含：

- **System Prompt**：定義 AI 的角色、專業規範、輸出要求
- **User Prompt Template**：定義如何組織參考案例與個案資料

#### 核心規範

1. **語言要求**：全程使用台灣繁體中文，嚴禁簡體字
2. **專業術語**：使用台灣職能治療專業術語，避免中國大陸用語
3. **分析邏輯**：模仿參考案例的臨床推理路徑
4. **輸出格式**：
   - 問題分析：條列式，格式為「[評估領域]：[具體描述]」
   - 治療建議：階層式條列，包含活動說明與目的

## 資料庫整合

此 skill 設計用於與以下資料庫配合：

- **向量資料庫**：ChromaDB (本地持久化)
- **Embedding 模型**：nomic-embed-text
- **生成模型**：qwen2.5:7b (或其他 Ollama 模型)

## 檢索策略

- **相似度閾值**：> 0.6
- **每區塊檢索數量**：前 3 筆最相關案例
- **區塊識別**：自動識別「精細動作」、「感覺統合」等評估領域

## 版本歷史

- **v1.0** (2026-02-04): 初始版本，包含標準報告生成功能
