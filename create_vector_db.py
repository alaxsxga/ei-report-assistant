import os
import json
import chromadb
from pathlib import Path
from typing import List, Dict, Any
import requests
from datetime import datetime

# =================設定區=================
# 向量資料庫儲存路徑 (會存在您的專案資料夾下)
DB_PATH = "./local_vector_db"
COLLECTION_NAME = "ot_reports"

# Ollama 設定
OLLAMA_BASE_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"  # 務必確認已執行 ollama pull nomic-embed-text
# =======================================

class LocalRAGBuilder:
    def __init__(self):
        print(f"初始化 ChromaDB (路徑: {DB_PATH})...")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
        # 建立或取得 Collection
        # 使用 cosine distance (適合語意搜尋)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"目前資料庫內已有 {self.collection.count()} 筆資料")

    def get_embedding(self, text: str) -> List[float]:
        """呼叫 Ollama 產生向量"""
        try:
            response = requests.post(
                OLLAMA_BASE_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"向量化失敗: {e}")
            print("請確認 Ollama 已啟動，且已下載模型: ollama pull nomic-embed-text")
            raise

    def process_json_to_chunks(self, data: Dict) -> List[Dict]:
        """將結構化 JSON 拆解為「因果連結」的完整語意塊"""
        chunks = []
        
        # 取得基本資訊
        child_info = data.get("child_info", {})
        name = child_info.get("name_or_id", "Unknown")
        age = child_info.get("age_at_assessment", "Unknown")
        source_file = data.get("source_file", "unknown")
        
        # 取得整體推理與建議 (這將作為每個塊的「答案」部分)
        analysis = data.get("problem_analysis_structured", {})
        reasoning = analysis.get("clinical_reasoning_text") or data.get("problem_analysis", "")
        impact = analysis.get("impact_on_function", "")
        issues = analysis.get("main_issues", [])
        
        recs = data.get("recommendations", {})
        goals = recs.get("treatment_goals", [])
        strategies = recs.get("home_school_strategies", [])
        activities = recs.get("suggested_activities", [])

        # 格式化建議與推理，作為共同的「邏輯結尾」
        logic_suffix = (
            f"\n--- 專業分析與建議核心 ---\n"
            f"【臨床推理與問題分析】：\n{reasoning}\n"
            f"【核心問題】：{', '.join(issues) if isinstance(issues, list) else issues}\n"
            f"【治療目標與課程重心】：{', '.join(goals) if isinstance(goals, list) else goals}\n"
            f"【具體建議與建議活動】：{', '.join(activities) if isinstance(activities, list) else activities}\n"
            f"【居家與學校策略建議】：{', '.join(strategies) if isinstance(strategies, list) else strategies}"
        )

        base_metadata = {
            "child_name": name,
            "child_age": age,
            "source_file": source_file,
            "processed_at": datetime.now().isoformat()
        }

        # 1. 強化後的領域塊 (核心：評估+建議 捆綁)
        domains = data.get("assessment_domains", [])
        for idx, domain in enumerate(domains):
            domain_name = domain.get("domain", "未分類")
            status = domain.get("status", "未知")
            obs = domain.get("qualitative_observation") or domain.get("observations", "")
            scores = domain.get("quantitative_data") or domain.get("scores", "")
            interp = domain.get("interpretation") or domain.get("findings", "")
            
            # 組合出一段「有因有果」的描述文字
            content = (
                f"【領域現狀】個案：{name}。評估領域：{domain_name}。狀態：{status}。\n"
                f"【觀察與表現】：{obs}\n"
                f"【數據與結果】：{scores}\n"
                f"【綜合解釋】：{interp}\n"
                f"{logic_suffix}"  # 強行接入該全案的推理與建議
            )
            
            chunks.append({
                "id": f"{source_file}_domain_{idx}",
                "text": content,
                "metadata": {
                    **base_metadata, 
                    "type": "assessment_domain",
                    "domain": domain_name,
                    "status": status
                }
            })

        # 2. 綜合描述塊 (針對主訴搜尋)
        concerns = data.get("family_concerns", [])
        concerns_text = "、".join(concerns) if isinstance(concerns, list) else str(concerns)
            
        chunks.append({
            "id": f"{source_file}_profile",
            "text": f"【個案主訴】姓名：{name}，年齡：{age}。主訴期待：{concerns_text}\n{logic_suffix}",
            "metadata": {**base_metadata, "type": "profile"}
        })
            
        return chunks

    def add_to_db(self, chunks: List[Dict]):
        """將處理好的塊存入資料庫"""
        if not chunks:
            return
            
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        print(f"  正在產生 {len(chunks)} 個向量 (使用 {EMBEDDING_MODEL})...")
        
        # 批次產生 embedding
        embeddings = []
        for text in texts:
            embeddings.append(self.get_embedding(text))
            
        # Upsert (如果 id 存在就更新，不然就新增)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"  ✓ 成功存入 {len(chunks)} 筆資料")


def main():
    print("="*60)
    print("建立 Local 向量知識庫 (ChromaDB + Ollama)")
    print("="*60)
    
    input_dir = Path("structured files")
    if not input_dir.exists():
        print(f"找不到輸入資料夾: {input_dir}")
        return

    # 1. 取得所有 json 檔
    json_files = list(input_dir.glob("*_structured.json"))
    if not json_files:
        print("沒有找到結構化 JSON 檔案 (.json)")
        return
        
    print(f"找到 {len(json_files)} 個檔案待處理")
    
    # 2. 初始化 builder
    try:
        builder = LocalRAGBuilder()
    except Exception as e:
        print(f"初始化失敗: {e}")
        return
        
    # 3. 處理每個檔案
    for i, fpath in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] 讀取: {fpath.name}")
        
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            chunks = builder.process_json_to_chunks(data)
            print(f"  拆解為 {len(chunks)} 個語意塊")
            
            builder.add_to_db(chunks)
            
        except Exception as e:
            print(f"  ✗ 處理失敗: {e}")

    print("\n" + "="*60)
    print("全部完成！向量資料庫已建立。")
    print(f"資料庫路徑: {os.path.abspath(DB_PATH)}")
    print("="*60)

if __name__ == "__main__":
    main()
