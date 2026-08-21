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
from prompts import (
    get_json_system_prompt, get_json_user_prompt,
    get_segmentation_system_prompt, get_segmentation_user_prompt
)

# ================= 設定區 =================
# 資料庫設定
DB_PATH = "./local_vector_db"
COLLECTION_NAME = "ot_reports"

# Ollama 設定 (用於 Embedding 和生成)
OLLAMA_API_URL = "http://localhost:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"  # 必須與建立資料庫時一致
GENERATION_MODEL = "gemma2"          # Google 開源模型，邏輯性強、回覆乾淨

# Step A（拆解區塊）固定用本地模型，不管使用者選哪個生成模型——
# 拆解區塊是範圍較窄的分類任務，本地小模型測試起來夠穩，且完全不受雲端 API 503／頻率限制影響
SEGMENTATION_MODEL_CHOICE = "Gemma2 (Local)"

# Anthropic 設定
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-5"

# Gemini 設定
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.7-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# =========================================

# 1. 資料庫連線函式
def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION_NAME)

def get_known_domains(collection):
    """取得資料庫裡實際存在的領域名稱清單"""
    data = collection.get(include=["metadatas"])
    return {m["domain"] for m in data.get("metadatas", []) if m.get("domain")}

def match_canonical_domains(label, known_domains):
    """把使用者輸入的區塊標籤對應回資料庫裡真實的領域名稱。
    完全相同的名稱跟「子分類」名稱（例如「日常生活自理」vs「日常生活自理－飲食」）都要一起找，
    不能只抓完全相同的就不找子分類了——不同案例可能用了不同細緻程度的領域命名。"""
    exact = [label] if label in known_domains else []
    contains = [d for d in known_domains if d != label and (label in d or d in label)]
    result = exact + contains
    return result or None

def call_llm_text(model_choice, system_prompt, user_prompt):
    """非串流呼叫，回傳完整文字（結構化 JSON 生成用，串流沒辦法邊收邊 parse JSON）。
    不自動重試——遇到雲端 API 暫時性錯誤（503 伺服器忙碌、429 頻率限制）直接拋出清楚的錯誤訊息，
    由使用者自行決定要不要重新送出。"""
    try:
        return _call_llm_text_once(model_choice, system_prompt, user_prompt)
    except Exception as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
        if status_code == 503:
            raise RuntimeError(f"{model_choice} 伺服器目前忙碌中（503），請稍後再試一次。") from e
        if status_code == 429:
            raise RuntimeError(f"{model_choice} 已達頻率限制（429），請稍等一下再試。") from e
        raise

def _call_llm_text_once(model_choice, system_prompt, user_prompt):
    if model_choice == "Claude Sonnet 5 (Cloud)":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,  # 領域數多時（結構化建議 JSON）很容易超過 4096 被截斷，parse 會直接失敗
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return next(block.text for block in message.content if block.type == "text")

    elif model_choice == "Gemini 3.7 Flash (Cloud)":
        url = f"{GEMINI_API_URL}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 16000}
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()
        return body["candidates"][0]["content"]["parts"][0]["text"]

    else:
        resp = requests.post(
            f"{OLLAMA_API_URL}/chat",
            json={
                "model": GENERATION_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "options": {"temperature": 0.2},
                "stream": False
            }
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

def parse_json_response(text):
    """去除 markdown 標記後 parse JSON"""
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return json.loads(t.strip())

def normalize_bullets(text):
    """統一「●」前面的換行格式，不依賴模型自己排版正確。
    不同模型（Gemini／Claude）在 JSON 字串裡放的換行符號不一定會被前端 Markdown 渲染成真的換行，
    這裡統一改成 CommonMark 的 hard break（兩個空白+換行），不管前端 markdown 引擎設定如何都會正確換行。"""
    if not text or "●" not in text:
        return text
    parts = re.split(r'\s*●', text)
    head = parts[0].strip()
    bullets = [p.strip() for p in parts[1:] if p.strip()]
    if not bullets:
        return text
    body = "  \n".join(f"●{b}" for b in bullets)
    return f"{head}\n\n{body}" if head else body

def segment_case_with_llm(case_description, model_choice, known_domains):
    """用 LLM 語意判讀拆分區塊——真實報告排版變化很多（有無冒號、括號編號、
    重複小標題），交給 LLM 對照已知領域清單來判讀。解析失敗就直接把錯誤往上拋，
    不要靜默退回品質差很多的規則比對，讓使用者在不知情的情況下拿到打折的結果。"""
    system_prompt = get_segmentation_system_prompt()
    user_prompt = get_segmentation_user_prompt(case_description, known_domains)
    raw = call_llm_text(model_choice, system_prompt, user_prompt)
    data = parse_json_response(raw)

    # 用 (d.get(key) or "") 而不是 d.get(key, "")——LLM 有時會把值明確設成 null 而不是省略欄位，
    # 這種情況 .get(key, "") 拿到的還是 None，不是預設值，直接 .strip() 會噴錯
    skipped = [d.get("domain") for d in data if d.get("domain") and not d.get("has_issue", True)]
    if skipped:
        print(f"⏭️ 判定為無異常/不需要，不列入報告：{skipped}")

    sections = [
        ((d.get("domain") or "").strip(), (d.get("content") or "").strip())
        for d in data
        if (d.get("domain") or "").strip() and (d.get("content") or "").strip() and d.get("has_issue", True)
    ]
    print(f"🧩 LLM 區塊解析完成：{[s[0] for s in sections]}" if sections else "🧩 LLM 判讀：沒有找到需要處理的問題領域")
    return sections

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
    if model_choice == "Claude Sonnet 5 (Cloud)":
        print(f"📝 使用 API 模型 ID: {CLAUDE_MODEL}")
    elif model_choice == "Gemini 3.7 Flash (Cloud)":
        print(f"📝 使用 API 模型 ID: {GEMINI_MODEL}")
    
    status_msg = "正在分析資料..."
    yield status_msg
    
    # --- 步驟 A: 解析與分割區塊（固定用本地模型，跟生成用的模型無關，見 SEGMENTATION_MODEL_CHOICE） ---
    collection = get_chroma_collection()
    known_domains = get_known_domains(collection)

    try:
        sections = segment_case_with_llm(case_description, SEGMENTATION_MODEL_CHOICE, known_domains)
    except Exception as e:
        print(f"❌ 區塊解析失敗: {e}")
        yield status_msg + f"\n❌ 區塊解析失敗：{e}"
        return
    print(f"📋 解析到內容區塊: {[s[0] for s in sections] if sections else '無(全域檢索)'}")

    if not sections:
        query_tasks = [("綜合描述", case_description)]
    else:
        query_tasks = [(s[0].strip(), s[1].strip()) for s in sections if s[1].strip()]

    problem_domain_context = {}  # 每個真實領域各自的參考資料，是結構化生成唯一的資料來源

    status_msg += f"\n檢測到 {len(query_tasks)} 個評估區塊，開始分區檢索..."
    yield status_msg

    # --- 步驟 B: 只針對「對應得到資料庫真實領域」的區塊做檢索 ---
    # 對不到領域的內容（例如「主訴」）不是評估領域，不參與檢索、也不會出現在最終報告裡
    for domain, content in query_tasks:
        matched_domains = match_canonical_domains(domain, known_domains)
        if not matched_domains:
            print(f"⏭️ 「{domain}」不是資料庫裡的評估領域，略過檢索")
            continue

        print(f"🔍 正在檢索領域: {domain}...")
        status_msg += f"\n🔍 檢索「{domain}」相關資料..."
        yield status_msg

        search_text = f"{domain}：{content}"
        embedding = get_embedding(search_text)
        if not embedding:
            print(f"❌ 「{domain}」Embedding 失敗")
            problem_domain_context[domain] = ""
            continue

        # 優先用「領域」metadata 鎖定範圍，避免被其他領域但字面相似的內容打敗
        domain_clause = (
            {"domain": matched_domains[0]}
            if len(matched_domains) == 1
            else {"domain": {"$in": matched_domains}}
        )
        # 領域內優先找「有建議內容」的案例（狀態異常、有問題分析），
        # 否則光靠 embedding 相似度容易撈到主題相近但狀態是「無異常」的案例，沒有建議可用
        where_with_rec = {"$and": [domain_clause, {"has_recommendation": True}]}
        results = collection.query(query_embeddings=[embedding], n_results=3, where=where_with_rec)
        if not (results['distances'] and results['distances'][0]):
            print(f"   ℹ️ 「{domain}」領域內沒有帶建議的案例，改抓一般觀察資料")
            results = collection.query(query_embeddings=[embedding], n_results=3, where=domain_clause)
        print(f"   🎯 鎖定領域：{matched_domains}")

        domain_docs = []
        if results['distances'] and results['distances'][0]:
            for i, dist in enumerate(results['distances'][0][:2]):  # 最多保留前 2 筆最相似的
                similarity = 1.0 - dist
                if similarity > 0.3:  # 領域已鎖定，門檻可放寬，只用來濾掉完全不相關的
                    domain_docs.append(results['documents'][0][i])
        problem_domain_context[domain] = "\n\n".join(domain_docs)
        print(f"✅ 「{domain}」檢索完成，找到 {len(domain_docs)} 筆相似資料")

    # --- 步驟 C: 生成 (Generation) ---
    print(f"🧠 準備進入 LLM 生成階段...")

    # 組出「真正對應到資料庫領域」的問題區塊清單（保證每個都有份）
    domain_blocks = [
        {"domain": domain, "case_issue": content, "reference": problem_domain_context[domain]}
        for domain, content in query_tasks
        if domain in problem_domain_context
    ]

    if domain_blocks:
        # 顯示面板要跟實際生成用的資料一致：直接顯示 domain_blocks 裡每個領域自己的 reference，
        # 不要再套用另一套「全域去重＋上限 5 筆」的邏輯（那套只在下面備援路徑才會被真的用到）
        found_blocks = [b for b in domain_blocks if b["reference"]]
        if not found_blocks:
            retrieval_info = "\n\n---\n## 📋 檢索結果\n\n未找到足夠相似的參考案例，各領域將保守生成。\n\n---\n"
        else:
            retrieval_info = "\n\n---\n## 📋 檢索結果\n\n"
            retrieval_info += f"**{len(found_blocks)}／{len(domain_blocks)} 個領域找到參考案例：**\n\n"
            for b in domain_blocks:
                retrieval_info += f"### 「{b['domain']}」\n\n"
                if b["reference"]:
                    preview = "\n".join(b["reference"].split("\n")[:5])
                    retrieval_info += f"```\n{preview}\n...\n```\n\n"
                else:
                    retrieval_info += "（沒有找到足夠相似的參考案例，此領域內容將較保守）\n\n"
            retrieval_info += "---\n\n## 🤖 開始生成報告...\n\n"

        # --- 結構化生成：LLM 只負責每個領域各自的內容，領域清單/編號/排版由程式碼保證完整 ---
        yield status_msg + retrieval_info + "\n🧠 正在針對各領域生成內容..."

        json_system_prompt = get_json_system_prompt()
        json_user_prompt = get_json_user_prompt(domain_blocks)

        try:
            raw = call_llm_text(model_choice, json_system_prompt, json_user_prompt)
            data = parse_json_response(raw)
        except Exception as e:
            print(f"❌ 結構化生成失敗: {e}")
            yield status_msg + retrieval_info + f"\n❌ 生成失敗：{e}"
            return

        result_domains = {d.get("domain"): d for d in data.get("domains", [])}
        expected = {b["domain"] for b in domain_blocks}
        missing = expected - set(result_domains.keys())

        for miss_domain in missing:
            print(f"⚠️ 「{miss_domain}」缺漏，補呼叫一次...")
            block = next(b for b in domain_blocks if b["domain"] == miss_domain)
            try:
                retry_raw = call_llm_text(model_choice, json_system_prompt, get_json_user_prompt([block]))
                retry_data = parse_json_response(retry_raw)
                for d in retry_data.get("domains", []):
                    result_domains[d.get("domain")] = d
            except Exception as e:
                print(f"   補呼叫失敗：{e}")

        still_missing = expected - set(result_domains.keys())
        if still_missing:
            print(f"⚠️ 補呼叫後仍缺漏：{still_missing}")

        # --- 組裝最終報告，領域清單由程式碼掌控，保證不會漏 ---
        # 一樣用 (d.get(key) or 預設值)，防止 LLM 把欄位明確設成 null 而不是省略或給空字串
        lines_out = ["### 問題分析"]
        for idx, b in enumerate(domain_blocks, 1):
            d = result_domains.get(b["domain"])
            issue = (d.get("issue_summary") if d else None) or b["case_issue"]
            lines_out.append(f"{idx}. {b['domain']}：{issue}")

        lines_out.append("")
        lines_out.append("### 總結與建議")
        lines_out.append(f"1. {data.get('course_recommendation') or '綜合以上結果，建議安排職能療育課程'}")
        lines_out.append("")
        for idx, b in enumerate(domain_blocks, 2):
            d = result_domains.get(b["domain"])
            rec = (d.get("recommendation") if d else None) or "（暫無足夠參考資料，建議由治療師進一步評估後補充）"
            lines_out.append(f"{idx}. {b['domain']}")
            lines_out.append("")
            lines_out.append(normalize_bullets(rec))
            lines_out.append("")

        print("✅ 結構化生成完畢")
        yield "\n".join(lines_out)

    else:
        # 輸入裡沒有任何內容能對應到資料庫的真實評估領域，沒有素材可以結構化生成，直接清楚告知，
        # 不再退回另一套「整段自由生成」的邏輯——只保留一條生成路徑。
        print("⚠️ 沒有找到可處理的評估領域內容")
        yield status_msg + "\n⚠️ 沒有找到可以處理的評估領域內容，請確認輸入內容是否包含實際的評估領域描述（例如：精細動作、感覺統合等）。"



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
                choices=["Gemma2 (Local)", "Gemini 3.7 Flash (Cloud)", "Claude Sonnet 5 (Cloud)"],
                value="Gemini 3.7 Flash (Cloud)",
                label="選擇生成模型"
            )
            api_key_input = gr.Textbox(
                label="API Key (若使用 Gemini 或 Claude)",
                placeholder="請輸入 API Key...",
                type="password",
                visible=True
            )
            
            def toggle_api_input(choice):
                if choice == "Gemma2 (Local)":
                    return gr.update(visible=False)
                return gr.update(visible=True)
                
            model_radio.change(fn=toggle_api_input, inputs=[model_radio], outputs=[api_key_input])

            btn_submit = gr.Button("🧠 開始生成報告", variant="primary")
            
        with gr.Column(scale=1):
            # 使用 Markdown 元件顯示，視覺效果最佳
            output_report = gr.Markdown(label="生成的報告內容")
            
    # 綁定事件
    def process_with_key(case, model, key):
        global ANTHROPIC_API_KEY, GEMINI_API_KEY
        if key:
            if "sk-" in key:
                ANTHROPIC_API_KEY = key
            else:
                GEMINI_API_KEY = key
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
