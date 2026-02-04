import gradio as gr
import chromadb
import requests
import json
import os
import re
import sys

# 加入 skill 路徑以便導入 prompt 模組
SKILL_PATH = os.path.join(os.path.dirname(__file__), '.agent', 'skills', 'ot-report-generation')
sys.path.insert(0, SKILL_PATH)

# 導入 prompt 模組
from prompts import get_system_prompt, get_user_prompt

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
    # 使用 skill 模組中的 prompt
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(context_str, case_description)

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
