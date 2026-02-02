import gradio as gr
import chromadb
import requests
import json
import os

# ================= 設定區 =================
# 資料庫設定
DB_PATH = "./local_vector_db"
COLLECTION_NAME = "ot_reports"

# Ollama 設定 (用於 Embedding 和生成)
OLLAMA_API_URL = "http://localhost:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"  # 必須與建立資料庫時一致
GENERATION_MODEL = "qwen2.5:7b"      # 生成用的模型，可換成 llama3 或其他
# =========================================

# 1. 資料庫連線函式
def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION_NAME)

# 2. Embedding 函式 (將文字轉向量)
def get_embedding(text):
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print(f"Embedding Error: {response.text}")
            return None
    except Exception as e:
        print(f"Ollama Connection Error: {e}")
        return None

# 3. 生成回應函式 (RAG 核心邏輯)
def generate_report(case_description):
    
    status_msg = "正在分析資料..."
    yield status_msg # 回傳給單一輸出欄位
    
    # 查詢文字直接使用輸入內容
    query_text = case_description
    
    # --- 步驟 A: 檢索 (Retrieval) ---
    embedding = get_embedding(query_text)
    if not embedding:
        err_msg = "錯誤：無法連接 Ollama，請確認 Ollama 已啟動。"
        yield err_msg
        return

    collection = get_chroma_collection()
    # 為了確保能捕捉到「評估工具」與「生活自理」的高度相關資料，先取前 5 筆再來過濾
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5 
    )
    
    status_msg += "\n已檢索到參考案例，正在進行關聯度篩選..."
    yield status_msg

    # 整理參考資料字串 (嚴格過濾：相似度 > 0.6)
    filtered_docs = []
    if results['distances'] and results['documents']:
        for i, dist in enumerate(results['distances'][0]):
            similarity = 1.0 - dist
            if similarity > 0.6:
                doc = results['documents'][0][i]
                filtered_docs.append(f"【參考案例 {i+1} (關聯度: {similarity:.2f})】\n{doc}\n")
    
    if not filtered_docs:
        context_str = "（⚠️ 警告：資料庫中未找到相似度 > 0.6 的高相關案例，以下報告將僅基於一般職能治療原則生成，可能不夠精準）"
        status_msg += "\n⚠️ 未找到高相關案例 (>0.6)，僅依一般邏輯生成。"
    else:
        context_str = "\n".join(filtered_docs)
        status_msg += f"\n篩選完成，找到 {len(filtered_docs)} 筆高相關案例 (>0.6)。生成報告中..."
    
    yield status_msg
        
    # --- 步驟 B: 生成 (Generation) ---
    system_prompt = """你是一位專業的職能治療師 (OT)。
你的任務是根據「使用者提供的主訴與觀察」，高度依賴「歷史案例的分析邏輯」與「資料庫詞彙」，撰寫一份專業的【問題分析】與【治療建議】。

請嚴格遵守以下規則：
1. **必須全程使用台灣繁體中文 (Traditional Chinese, Taiwan)。**
2. **名詞規範**：專有名詞請嚴格使用參考案例中出現的資料庫名詞 (例如：本體覺、觸覺防禦、精細動作)，不要自行創造或使用不熟悉的別名。
3. **分析邏輯**：請模仿參考案例的「臨床推理路徑」，不要憑空發揮。例如：參考案例如何將「坐不住」連結到「前庭覺」，你就要沿用此邏輯。
4. **建議限制**：產出的建議內容與語意，請勿偏離資料庫資料的範疇。
5. 直接輸出報告內容，不要有開場白或結語。"""

    user_prompt = f"""
請參考以下「高度相關」的歷史案例（若無相關案例則請保守分析）：
{context_str}

--------------------------------------------------
【目前個案資料】
{case_description}

請根據上述參考資料的邏輯，撰寫：
1. ### 問題分析 (請使用資料庫中的分析邏輯，並以「條列式」呈現，敘述請精簡扼要)
2. ### 總結與建議 (第一點務必為「療育課程建議」，請模仿資料庫語氣，例如：「綜合以上結果，建議持續職能療育課程」。後續請列出各細項建議，並深入參考資料庫內容，敘述務必詳盡具體，避免過於簡略)
"""

    # 呼叫 Ollama 生成 (支援串流顯示)
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/chat",
            json={
                "model": GENERATION_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": True # 開啟串流
            },
            stream=True
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                # 必須先 decode byte string
                decoded_line = line.decode('utf-8')
                try:
                    body = json.loads(decoded_line)
                    if "message" in body:
                        token = body["message"]["content"]
                        full_response += token
                        yield full_response # 更新輸出
                    if body.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        err = f"生成時發生錯誤: {e}"
        yield err

# ================= 介面設計 (Gradio) =================
with gr.Blocks(title="AI 職能治療報告助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 AI 職能治療報告助手 (Local RAG)")
    gr.Markdown("輸入個案的主訴與觀察，將參考歷史病歷庫，生成問題分析與建議。")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_case = gr.Textbox(
                label="主訴與評估內容", 
                placeholder="例如：家屬表示孩子在學校坐不住，寫字很醜... 觀察發現抓握姿勢不成熟，無法單腳站立...",
                lines=15
            )
            btn_submit = gr.Button("🧠 開始生成報告", variant="primary")
            
        with gr.Column(scale=1):
            # 使用 Markdown 元件顯示，視覺效果最佳
            output_report = gr.Markdown(label="生成的報告內容")
            
    # 綁定事件
    btn_submit.click(
        fn=generate_report,
        inputs=[input_case],
        outputs=[output_report]
    )

if __name__ == "__main__":
    print("啟動網頁介面...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
