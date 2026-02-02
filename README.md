# AI 職能治療報告助手 (AI Occupational Therapy Assistant)

這是一套基於 **Local RAG (Retrieval-Augmented Generation)** 的智慧化輔助系統，專門設計用於協助職能治療師 (OT) 撰寫評估報告。系統能從過往的紙本/PDF 報告中學習分析邏輯，並根據新個案的主訴與觀察，自動生成專業的「問題分析」與「治療建議」。

## 🚀 核心功能

1.  **結構化資料萃取 (ETL)**：
    *   自動讀取 PDF/TXT 格式的原始評估報告。
    *   使用 AI (Claude/Ollama) 將非結構化文字轉換為標準化的 JSON 資料（含主訴、評估數據、問題分析、建議）。
2.  **臨床知識庫建立 (Indexing)**：
    *   將結構化資料進行語意分塊 (Chunking)。
    *   使用 Ollama embedding 模型將資料向量化，存入本地 ChromaDB 資料庫。
3.  **智慧報告生成 (Retrieval & Generation)**：
    *   提供 Web 介面 (Gradio)，輸入個案狀況即可檢索相似歷史案例。
    *   AI 嚴格遵循歷史案例的「臨床推理邏輯」與「專業術語」，生成高度擬真的報告草稿。

## 🛠️ 環境需求

*   **Python**: 3.11+
*   **Ollama**: 需安裝並在背景執行
    *   Embedding Model: `nomic-embed-text`
    *   Generation Model: `qwen2.5:7b` (或其他支援繁體中文的模型)
*   **主要 Python 套件**: `gradio`, `chromadb`, `anthropic` (若使用 Claude), `pdfplumber`

## 📂 專案結構

- **`extract_report.py`**: 資料處理核心。負責讀取 `raw files/` 中的 PDF，呼叫 AI 進行結構化萃取，並存入 `structured files/`。
- **`create_vector_db.py`**: 知識庫建置。讀取 `structured files/` 的 JSON，轉向量並存入 `./local_vector_db`。
- **`app.py`**: Web 應用程式。啟動 Gradio 使用者介面，執行 RAG 搜尋與報告生成。
- **`test_query.py`**: 測試腳本。用於測試向量資料庫的搜尋品質。
- **`raw files/`**: (資料夾) 存放原始 PDF 評估報告。
- **`structured files/`**: (資料夾) 存放處理後的 JSON 檔。

## ⚡️ 快速開始 (Quick Start)

### 1. 準備環境與模型
確保 Ollama 已啟動，並下載必要模型：
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

### 2. 資料前處理 (ETL)
將您的 PDF 報告放入 `raw files` 資料夾，然後執行：
```bash
# 預設使用 Claude 進行高精確度萃取 (需設定 API Key)，也可改用 Ollama
python3 extract_report.py
```
> 產出的 JSON 會存放在 `structured files` 資料夾。

### 3. 建立向量知識庫
將處理好的 JSON 資料寫入向量資料庫：
```bash
python3 create_vector_db.py
```

### 4. 啟動 AI 助手
開啟網頁介面開始使用：
```bash
python3 app.py
```
打開瀏覽器訪問 `http://localhost:7860` 即可。

## ⚙️ 進階設定
*   **切換模型**：在 `app.py` 中修改 `GENERATION_MODEL` 變數即可更換生成的 LLM。
*   **調整嚴格度**：`app.py` 中的 `similarity > 0.6` 門檻決定了參考資料的品質，可視需求調整。
