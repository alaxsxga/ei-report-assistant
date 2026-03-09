import gradio as gr
import chromadb
import requests
import json
import os
import re
import sys
import anthropic
import base64
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

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
GENERATION_MODEL = "gemma2"          # Google 開源模型，邏輯性強、回覆乾淨

# Anthropic 設定
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
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
def generate_report(case_description, model_choice):
    print(f"\n{'='*30}")
    print(f"🚀 開始生成報告任務")
    print(f"🤖 選擇模型: {model_choice}")
    if model_choice == "Claude 4 Sonnet (Cloud)":
        print(f"📝 使用 API 模型 ID: {CLAUDE_MODEL}")
    
    status_msg = "正在分析資料..."
    yield status_msg
    
    # --- 步驟 A: 解析與分割區塊 ---
    section_pattern = r'(?:\n|^)(?:\d+[\.、]\s*)?([\u4e00-\u9fa5]{2,6})(?:[:：]|\s|\n)([\s\S]*?)(?=(?:\n\d+[\.、]\s*|[\u4e00-\u9fa5]{2,6}[:：]|$))'
    sections = re.findall(section_pattern, case_description)
    print(f"📋 解析到內容區塊: {[s[0] for s in sections] if sections else '無(全域檢索)'}")
    
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
        print(f"🔍 正在檢索領域: {domain}...")
        status_msg += f"\n🔍 檢索「{domain}」相關資料..."
        yield status_msg
        
        search_text = f"{domain}：{content}"
        embedding = get_embedding(search_text)
        
        if not embedding:
            print(f"❌ 「{domain}」Embedding 失敗")
            continue
            
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3 
        )
        
        if results['distances'] and results['documents']:
            found_count = 0
            for i, dist in enumerate(results['distances'][0]):
                similarity = 1.0 - dist
                if similarity > 0.6:
                    found_count += 1
                    doc = results['documents'][0][i]
                    all_context_list.append(f"【針對「{domain}」的歷史參考資料 ({similarity:.2f})】\n{doc}\n")
            print(f"✅ 「{domain}」檢索完成，找到 {found_count} 筆相似資料")
    
    
    if not all_context_list:
        context_str = "（⚠️ 警告：所有區塊均未找到相似度 > 0.6 的案例，以下報告將僅基於一般邏輯生成）"
        status_msg += "\n⚠️ 未找到高相關案例。"
        retrieval_info = "\n\n---\n## 📋 檢索結果\n\n未找到相似度 > 0.6 的參考案例。\n\n---\n"
    else:
        context_str = "\n".join(all_context_list)
        status_msg += f"\n分區檢索完成，共收集 {len(all_context_list)} 筆高相關參考資料。"
        
        # 建立檢索結果摘要
        retrieval_info = "\n\n---\n## 📋 檢索結果\n\n"
        retrieval_info += f"**共檢索到 {len(all_context_list)} 筆參考資料：**\n\n"
        
        for idx, context in enumerate(all_context_list, 1):
            # 提取領域和相似度
            lines = context.split('\n')
            header = lines[0] if lines else ""
            preview = '\n'.join(lines[1:6]) if len(lines) > 1 else ""  # 顯示前5行
            
            retrieval_info += f"### {idx}. {header}\n\n"
            retrieval_info += f"```\n{preview}\n...\n```\n\n"
        
        retrieval_info += "---\n\n## 🤖 開始生成報告...\n\n"
    
    yield status_msg + retrieval_info
        
    yield status_msg + retrieval_info
        
    # --- 步驟 C: 生成 (Generation) ---
    print(f"🧠 準備進入 LLM 生成階段...")
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(context_str, case_description)

    if model_choice == "Claude 4 Sonnet (Cloud)":
        print(f"☁️ 正在呼叫 Anthropic Claude API...")
        # 使用 Anthropic Claude 生成
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            ) as stream:
                print("📝 Claude 串流開始接收...")
                full_response = ""
                for text in stream.text_stream:
                    full_response += text
                    yield full_response
                print("✅ Claude 生成完畢")
        except Exception as e:
            error_msg = f"Claude API 錯誤: {str(e)}"
            print(f"❌ {error_msg}")
            yield error_msg
    else:
        print(f"🏠 正在呼叫本地 Ollama ({GENERATION_MODEL})...")
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
                    decoded_line = line.decode('utf-8')
                    try:
                        body = json.loads(decoded_line)
                        if "message" in body:
                            token = body["message"]["content"]
                            full_response += token
                            yield full_response
                        if body.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            print("✅ Ollama 生成完畢")
        except Exception as e:
            error_msg = f"Ollama 生成時發生錯誤: {str(e)}"
            print(f"❌ {error_msg}")
            yield error_msg


# ================= 介面設計 (Gradio) =================

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# 定義極簡毛玻璃風格 (Minimalist Glassmorphism - Spring Edition)
custom_css = """
    /* 背景與基礎重設 */
    .gradio-container {
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
        color: #2d3748;
        background: linear-gradient(120deg, #f0fff4 0%, #fff5f0 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* 建立 Mesh Gradient 視覺效果 */
    .gradio-container::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: 
            radial-gradient(at 0% 0%, rgba(198, 246, 213, 0.6) 0, transparent 50%),
            radial-gradient(at 50% 0%, rgba(255, 239, 213, 0.6) 0, transparent 50%),
            radial-gradient(at 100% 0%, rgba(254, 215, 226, 0.5) 0, transparent 50%),
            radial-gradient(at 0% 100%, rgba(154, 230, 180, 0.4) 0, transparent 50%),
            radial-gradient(at 100% 100%, rgba(255, 226, 194, 0.5) 0, transparent 50%);
        z-index: -1;
    }

    /* 移除 Gradio 預設的深灰色背景與邊框 */
    #root, .main, .wrap, .cont, .form, .gr-form, .gr-padded, .padded {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }

    /* 強制所有區塊（卡片）保持一致的毛玻璃風格 */
    .block, .gr-box, .gr-panel, .form, fieldset {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(25px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(25px) saturate(160%) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 24px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.04) !important;
        padding: 30px !important;
        margin-bottom: 25px !important;
    }
    
    /* 針對 Radio 選項容器做特別處理，避免出現預設灰色 */
    .gr-radio-group, .wrap.inline {
        background: transparent !important;
        border: none !important;
    }
    
    /* 選項按鈕內部的容器 */
    .gr-input-label {
        background: rgba(255, 255, 255, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 12px !important;
        margin: 5px !important;
        transition: all 0.2s ease;
    }

    /* 標題 - 溫柔通透感 */
    h1 {
        color: #4a5568 !important;
        font-weight: 800 !important;
        font-size: 2.5em !important;
        text-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin: 0.5em 0 !important;
        text-align: center;
    }
    
    /* 副標題說明文字 */
    .gradio-container .prose p {
        color: #718096 !important;
        font-weight: 600;
        font-size: 1.1em;
        text-align: center;
    }

    /* 按鈕 - 粉橘漸層 */
    button.primary {
        background: linear-gradient(135deg, #f6ad55, #ed8936) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 17px !important;
        height: 54px !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 20px rgba(237, 137, 54, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1);
    }

    button.primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(237, 137, 54, 0.3) !important;
        filter: brightness(1.05);
    }

    /* Radio 選中狀態 - 淺綠主題 */
    .selected {
        background: rgba(72, 187, 120, 0.15) !important;
        color: #38a169 !important;
        border-color: #48bb78 !important;
    }

    /* 標籤文字 */
    span.label, label span, .meta-text {
        color: #4a5568 !important;
        font-weight: 700;
        margin-bottom: 12px;
        text-transform: uppercase;
        font-size: 13px;
        letter-spacing: 0.5px;
    }

    /* 文字輸入框樣式優化 */
    textarea {
        background: rgba(255, 255, 255, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 16px !important;
    }

    /* 卷軸美化 */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }
"""


with gr.Blocks(title="AI 職能治療報告助手") as demo:
    gr.Markdown("# 🏥 AI 職能治療報告助手 (Local RAG)\n輸入個案的主訴與觀察，將參考歷史病歷庫，生成問題分析與建議。")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_case = gr.Textbox(
                label="主訴與評估內容", 
                placeholder="例如：家屬表示孩子在學校坐不住，寫字很醜... 觀察發現抓握姿勢不成熟，無法單腳站立...",
                lines=12
            )
            model_radio = gr.Radio(
                choices=["Gemma2 (Local)", "Claude 4 Sonnet (Cloud)"],
                value="Claude 4 Sonnet (Cloud)",
                label="選擇生成模型"
            )
            api_key_input = gr.Textbox(
                label="Anthropic API Key (若使用 Claude)",
                placeholder="sk-...",
                type="password",
                visible=False
            )
            
            def toggle_api_input(choice):
                if choice == "Claude 3.5 Sonnet (Cloud)":
                    return gr.update(visible=True)
                return gr.update(visible=False)
                
            model_radio.change(fn=toggle_api_input, inputs=[model_radio], outputs=[api_key_input])

            btn_submit = gr.Button("🧠 開始生成報告", variant="primary")
            
        with gr.Column(scale=1):
            # 使用 Markdown 元件顯示，視覺效果最佳
            output_report = gr.Markdown(label="生成的報告內容")
            
    # 綁定事件
    def process_with_key(case, model, key):
        global ANTHROPIC_API_KEY
        if key:
            ANTHROPIC_API_KEY = key
        yield from generate_report(case, model)

    btn_submit.click(
        fn=lambda: gr.update(interactive=False, value="⏳ 正在生成報告..."),
        outputs=[btn_submit]
    ).then(
        fn=process_with_key,
        inputs=[input_case, model_radio, api_key_input],
        outputs=[output_report]
    ).then(
        fn=lambda: gr.update(interactive=True, value="🧠 開始生成報告"),
        outputs=[btn_submit]
    )

if __name__ == "__main__":
    print("啟動網頁介面...")
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Base(), css=custom_css)
