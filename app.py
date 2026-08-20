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
    get_system_prompt, get_user_prompt,
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

# Anthropic 設定
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-5"

# Gemini 設定
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
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

def parse_case_sections(case_description):
    """逐行解析「領域：內容」格式的區塊，取代舊版一次性 regex（會漏掉相鄰區塊）。
    支援同一個標籤底下有多行內容，直到遇到下一個「標籤：」開頭的行為止。"""
    line_pattern = re.compile(r'^\s*(?:\d+[\.、]\s*)?([一-龥A-Za-z0-9]{2,12})[:：]\s*(.*)$')
    sections = []
    current_label = None
    current_lines = []

    def flush():
        if current_label is not None:
            text = '\n'.join(current_lines).strip()
            if text:
                sections.append((current_label, text))

    for line in case_description.split('\n'):
        m = line_pattern.match(line)
        if m:
            flush()
            current_label = m.group(1).strip()
            current_lines = [m.group(2)]
        elif current_label is not None:
            current_lines.append(line)
    flush()
    return sections

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
    if model_choice == "Claude 4 Sonnet (Cloud)":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,  # 領域數多時（結構化建議 JSON）很容易超過 4096 被截斷，parse 會直接失敗
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return next(block.text for block in message.content if block.type == "text")

    elif model_choice == "Gemini 2.5 Flash (Cloud)":
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
    """用 LLM 語意判讀拆分區塊，取代不可靠的 regex 規則比對——
    真實報告排版變化很多（有無冒號、括號編號 vs 數字編號、重複小標題），
    regex 永遠會漏掉某種寫法，交給 LLM 對照已知領域清單來判讀比較穩。"""
    try:
        system_prompt = get_segmentation_system_prompt()
        user_prompt = get_segmentation_user_prompt(case_description, known_domains)
        raw = call_llm_text(model_choice, system_prompt, user_prompt)
        data = parse_json_response(raw)

        skipped = [d["domain"] for d in data if d.get("domain") and not d.get("has_issue", True)]
        if skipped:
            print(f"⏭️ 判定為無異常/不需要，不列入報告：{skipped}")

        sections = [
            (d["domain"].strip(), d["content"].strip())
            for d in data
            if d.get("domain") and d.get("content", "").strip() and d.get("has_issue", True)
        ]
        if sections:
            print(f"🧩 LLM 區塊解析成功：{[s[0] for s in sections]}")
            return sections
    except Exception as e:
        print(f"⚠️ LLM 區塊解析失敗，改用規則解析: {e}")

    return parse_case_sections(case_description)

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

def generate_freeform_stream(model_choice, system_prompt, user_prompt):
    """舊版：整段自由生成、邊生成邊串流。當輸入無法拆成領域區塊（沒有 domain_blocks）時的備用路徑，
    無法保證每個領域都不會被漏寫，但至少能對非結構化輸入給出合理結果。"""
    if model_choice == "Claude 4 Sonnet (Cloud)":
        print(f"☁️ 正在呼叫 Anthropic Claude API...")
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
            error_msg = f"❌ Claude API 錯誤: {str(e)}"
            print(error_msg)
            yield error_msg
    elif model_choice == "Gemini 2.5 Flash (Cloud)":
        print(f"☁️ 正在呼叫 Google Gemini API...")
        if not GEMINI_API_KEY:
            yield "❌ 錯誤：未偵測到 Gemini API Key，請檢查輸入或 .env 內容。"
            return

        try:
            # 加上 alt=sse 參數，強制回傳標準 Server-Sent Events (SSE) 單行 JSON 格式，避免切割換行錯誤
            url = f"{GEMINI_API_URL}/{GEMINI_MODEL}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"

            # 使用正確的 v1beta 支援格式：system_instruction 必須放在最外層
            payload = {
                "system_instruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 20000,
                }
            }

            print(f"📡 發送請求至: {url}")
            response = requests.post(url, json=payload, stream=True)

            if response.status_code != 200:
                try:
                    error_data = response.json()
                    print(f"DEBUG: API Error Raw Data: {error_data}")

                    # 處理不同的錯誤格式 (dict 或 list)
                    if isinstance(error_data, list):
                        error_data = error_data[0]

                    error_obj = error_data.get('error', {}) if isinstance(error_data, dict) else {}
                    error_msg_detail = error_obj.get('message', str(error_data))
                    error_msg = f"❌ Gemini API 伺服器回報錯誤 ({response.status_code})：{error_msg_detail}"
                except Exception:
                    error_msg = f"❌ Gemini API 伺服器回報錯誤 ({response.status_code})，且無法解析內容。原始回應：{response.text[:200]}"

                print(error_msg)
                yield error_msg
                return

            full_response = ""
            print("📝 Gemini 串流開始接收(SSE 模式)...")
            for line in response.iter_lines():
                if not line: continue
                decoded_line = line.decode('utf-8').strip()

                # 處理標準 SSE 格式
                if decoded_line.startswith("data: "):
                    json_str = decoded_line[6:].strip() # 取出 "data: " 後面的內容
                    if not json_str or json_str == "[DONE]":
                        continue

                    try:
                        body = json.loads(json_str)

                        if "candidates" in body and body["candidates"]:
                            candidate = body["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                parts = candidate["content"].get("parts", [])
                                if parts:
                                    part = parts[0]
                                    if isinstance(part, dict) and "text" in part:
                                        token = part.get("text", "")
                                        full_response += token
                                        yield full_response

                            # 加入安全診斷與錯誤原因印出
                            finish_reason = candidate.get("finishReason")
                            if finish_reason and finish_reason != "STOP":
                                safety_msg = f"\n⚠️ [警告] Gemini 生成被迫中止。原因: {finish_reason}"
                                print(safety_msg)
                                if "safetyRatings" in candidate:
                                    print(f"DEBUG: 觸發的安全過濾: {candidate.get('safetyRatings')}")
                                yield full_response + safety_msg
                        else:
                            print(f"DEBUG: 發生預期外結構: {body}")

                    except Exception as ex:
                        print(f"⚠️ 解析 JSON 發生錯誤: {ex} 原始字串: {json_str[:50]}...")
                        continue

            if not full_response:
                yield "⚠️ Gemini 回傳內容為空。這通常是因為「醫療字眼」觸發了 Google 的安全防護 (Safety Filter) 被截斷，建議查閱終端機日誌。"
            else:
                print(f"✅ Gemini 生成完畢，共 {len(full_response)} 字")

        except Exception as e:
            error_msg = f"❌ Gemini API 調用發生嚴重異常: {str(e)}"
            print(error_msg)
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
                    "options": {"temperature": 0.2},
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
            error_msg = f"❌ Ollama 生成時發生錯誤: {str(e)}"
            print(error_msg)
            yield error_msg

# 3. 生成回應函式 (RAG 核心邏輯)
def generate_report(case_description, model_choice):
    print(f"\n{'='*30}")
    print(f"🚀 開始生成報告任務")
    print(f"🤖 選擇模型: {model_choice}")
    if model_choice == "Claude 4 Sonnet (Cloud)":
        print(f"📝 使用 API 模型 ID: {CLAUDE_MODEL}")
    elif model_choice == "Gemini 2.5 Flash (Cloud)":
        print(f"📝 使用 API 模型 ID: {GEMINI_MODEL}")
    
    status_msg = "正在分析資料..."
    yield status_msg
    
    # --- 步驟 A: 解析與分割區塊（用 LLM 語意判讀，比 regex 更能處理真實報告排版） ---
    collection = get_chroma_collection()
    known_domains = get_known_domains(collection)

    sections = segment_case_with_llm(case_description, model_choice, known_domains)
    print(f"📋 解析到內容區塊: {[s[0] for s in sections] if sections else '無(全域檢索)'}")

    if not sections:
        query_tasks = [("綜合描述", case_description)]
    else:
        query_tasks = [(s[0].strip(), s[1].strip()) for s in sections if s[1].strip()]

    all_candidates = []
    problem_domain_context = {}  # 只收「有對應到真實領域」的區塊，給結構化生成用（保證每個都有份）

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

        # 優先用「領域」metadata 鎖定範圍，避免被其他領域但字面相似的內容打敗
        matched_domains = match_canonical_domains(domain, known_domains)
        if matched_domains:
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
            similarity_floor = 0.3  # 領域已鎖定，門檻可放寬
            print(f"   🎯 鎖定領域：{matched_domains}")

            # 這個領域自己的參考資料（不受下面全域去重/上限影響，保證每個真實領域都有份）
            if results['distances'] and results['distances'][0]:
                domain_docs = [
                    results['documents'][0][i]
                    for i in range(min(2, len(results['distances'][0])))
                ]
                problem_domain_context[domain] = "\n\n".join(domain_docs)
            else:
                problem_domain_context[domain] = ""
        else:
            results = collection.query(query_embeddings=[embedding], n_results=3)
            similarity_floor = 0.6  # 沒對應到領域，維持全庫比對的高門檻
            print(f"   🌐 無對應領域，改用全庫比對")

        if results['distances'] and results['documents'] and results['distances'][0]:
            found_count = 0
            for i, dist in enumerate(results['distances'][0]):
                similarity = 1.0 - dist
                if similarity > similarity_floor and found_count < 2:  # 每區塊最多保留前 2 筆最相似的
                    found_count += 1
                    doc = results['documents'][0][i]
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    source_file = metadata.get('source_file') or f"{domain}_{i}"
                    all_candidates.append({
                        "domain": domain,
                        "similarity": similarity,
                        "doc": doc,
                        "source_file": source_file
                    })
            print(f"✅ 「{domain}」檢索完成，找到 {found_count} 筆相似資料")

    # --- 跨區塊去重與總量上限，避免內容線性膨脹 ---
    # 同一份歷史案例（source_file 相同）只保留相似度最高的一筆，其餘視為重複內容濾掉
    best_by_source = {}
    for c in all_candidates:
        key = c["source_file"]
        if key not in best_by_source or c["similarity"] > best_by_source[key]["similarity"]:
            best_by_source[key] = c

    MAX_CONTEXT_CHUNKS = 5  # 全部區塊合計最多取前 5 筆最相似的參考資料
    deduped_candidates = sorted(best_by_source.values(), key=lambda c: c["similarity"], reverse=True)[:MAX_CONTEXT_CHUNKS]

    all_context_list = [
        f"【針對「{c['domain']}」的歷史參考資料 ({c['similarity']:.2f})】\n{c['doc']}\n"
        for c in deduped_candidates
    ]
    if len(all_candidates) > len(all_context_list):
        print(f"🧹 過濾：候選 {len(all_candidates)} 筆 → 去重/上限後保留 {len(all_context_list)} 筆")


    if not all_context_list:
        context_str = "（⚠️ 警告：所有區塊均未找到足夠相似的案例，以下報告將僅基於一般邏輯生成）"
        status_msg += "\n⚠️ 未找到高相關案例。"
        retrieval_info = "\n\n---\n## 📋 檢索結果\n\n未找到足夠相似的參考案例。\n\n---\n"
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

    # 組出「真正對應到資料庫領域」的問題區塊清單（保證每個都有份，不受前面全域去重/上限影響）
    domain_blocks = [
        {"domain": domain, "case_issue": content, "reference": problem_domain_context[domain]}
        for domain, content in query_tasks
        if domain in problem_domain_context
    ]

    if domain_blocks:
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
        lines_out = ["### 問題分析"]
        for idx, b in enumerate(domain_blocks, 1):
            d = result_domains.get(b["domain"])
            issue = d.get("issue_summary") if d else b["case_issue"]
            lines_out.append(f"{idx}. {b['domain']}：{issue}")

        lines_out.append("")
        lines_out.append("### 總結與建議")
        lines_out.append(f"1. {data.get('course_recommendation', '綜合以上結果，建議安排職能療育課程')}")
        lines_out.append("")
        for idx, b in enumerate(domain_blocks, 2):
            d = result_domains.get(b["domain"])
            rec = d.get("recommendation") if d else "（暫無足夠參考資料，建議由治療師進一步評估後補充）"
            lines_out.append(f"{idx}. {b['domain']}")
            lines_out.append("")
            lines_out.append(normalize_bullets(rec))
            lines_out.append("")

        print("✅ 結構化生成完畢")
        yield "\n".join(lines_out)

    else:
        # --- 備用路徑：輸入無法拆成領域區塊（例如整段沒有「領域：」格式），退回舊版整段自由生成 ---
        print("ℹ️ 沒有對應到領域的區塊，改用整段自由生成（無法保證領域完整性）")
        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(context_str, case_description)
        yield from generate_freeform_stream(model_choice, system_prompt, user_prompt)



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
                choices=["Gemma2 (Local)", "Gemini 2.5 Flash (Cloud)", "Claude 4 Sonnet (Cloud)"],
                value="Gemini 2.5 Flash (Cloud)",
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
