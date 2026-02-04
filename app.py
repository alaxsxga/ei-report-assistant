import gradio as gr
import chromadb
import requests
import json
import os
import re

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
    
    # --- 步驟 A: 解析與分割區塊 (Decomposed RAG) ---
    # 匹配 "1. 精細動作" 或 "精細動作：" 等格式
    section_pattern = r'(?:\n|^)(?:\d+[\.、]\s*)?([\u4e00-\u9fa5]{2,6})(?:[:：]|\s|\n)([\s\S]*?)(?=(?:\n\d+[\.、]\s*|[\u4e00-\u9fa5]{2,6}[:：]|$))'
    sections = re.findall(section_pattern, case_description)
    
    if not sections:
        query_tasks = [("綜合描述", case_description)]
    else:
        query_tasks = [(s[0].strip(), s[1].strip()) for s in sections if s[1].strip()]

    collection = get_chroma_collection()
    all_context_list = []
    
    status_msg += f"\n檢測到 {len(query_tasks)} 個評估區塊，開始分區檢索..."
    yield status_msg

    # --- 步驟 B: 針對每個區塊進行個別檢索 ---
    for domain, content in query_tasks:
        status_msg += f"\n🔍 檢索「{domain}」相關資料..."
        yield status_msg
        
        # 將領域與內容合在一起做向量化，增加搜尋精準度
        search_text = f"{domain}：{content}"
        embedding = get_embedding(search_text)
        
        if not embedding:
            continue

        # 針對單一領域取相似度最高的前 3 筆，避免 Token 爆炸
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3 
        )
        
        if results['distances'] and results['documents']:
            for i, dist in enumerate(results['distances'][0]):
                similarity = 1.0 - dist
                if similarity > 0.6:
                    doc = results['documents'][0][i]
                    all_context_list.append(f"【針對「{domain}」的歷史參考資料 ({similarity:.2f})】\n{doc}\n")
    
    if not all_context_list:
        context_str = "（⚠️ 警告：所有區塊均未找到相似度 > 0.6 的案例，以下報告將僅基於一般邏輯生成）"
        status_msg += "\n⚠️ 未找到高相關案例。"
    else:
        context_str = "\n".join(all_context_list)
        status_msg += f"\n分區檢索完成，共收集 {len(all_context_list)} 筆高相關參考資料。生成報告中..."
    
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
1. **呈現方式**：大項目請使用條列式（如 1. 2. 3.）進行列點。
2. ### 問題分析
   - **核心問題模仿**：請從參考案例中找到「核心問題：」這一行，並**高度模仿**其記述模式。
   - **格式規範**：每一條列點必須嚴格遵循『**[評估领域名稱]：[具體功能受限或行為描述]**』的格式（例如：『1. 精細動作：工具使用經驗不足，運筆技巧處初階階段』）。
   - **禁止歸納與分析**：嚴禁寫成「...發展不足」或「...失調」等總結性結論。請直接將個案當下的具體行為表現填充進上述格式中。

3. ### 總結與建議
   - **階層規範**：**所有主項目的數字標號（1. 2. 3. ...）必須位於同一層級，嚴禁將第 2、3 點縮排在第 1 點內。** 每一個數字標號都必須另起新行並靠左對齊。僅在單一項目內部的詳細子項才使用符號（如 ● 或 - ）進行縮排。
   - **標號邏輯**：
    *   **1.** 固定為療育課程建議句型，請嚴格模仿資料庫中的句型模式（例如：『綜合以上結果，建議安排職能療育課程』或『建議安排短期小團體職能療育課程』等），僅輸出該固定語句，不可添加額外說明。
    *   **2. 之後**：直接延續數字標號，每一點對應一個問題領域。
   - **實作深度與專業溫度**：活動說明應詳盡且具備臨床同理心，避免生硬的指令與過度感性的文學描述。
      *   **敘述長度**：請詳盡描述活動的具體執行方式與家長可觀察的重點，內容應紮實且具備指導價值。
      *   **臨床同理心**：語氣應專業且穩定，從孩子的生活功能出發，說明活動如何減少其在日常生活中的挫折感。
      *   **目的明確**：必須清晰連結活動與改善目標的原因。
      *   **範例風格**：『建議透過觸覺探索活動（如在沙中尋寶）來降低觸覺防禦。執行時請觀察孩子對不同質地的接受度，透過漸進式的參與來提升其環境適應力，進而減少其在學校團體生活中的排斥感與情緒波動。』
      *   **範例結構**：
          1. 綜合以上結果，建議安排職能療育課程。
          2. 精細動作：......
          3. 感覺統合：......
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
        fn=lambda: gr.update(interactive=False, value="⏳ 正在生成報告..."),
        outputs=[btn_submit]
    ).then(
        fn=generate_report,
        inputs=[input_case],
        outputs=[output_report]
    ).then(
        fn=lambda: gr.update(interactive=True, value="🧠 開始生成報告"),
        outputs=[btn_submit]
    )

if __name__ == "__main__":
    print("啟動網頁介面...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
