"""
早療評估報告資料預處理工具 - 職能治療專用版
針對職能治療評估報告優化的結構化分析
"""

import os
import json
from pathlib import Path
from typing import Dict
import anthropic
from datetime import datetime

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class OccupationalTherapyReportProcessor:
    """職能治療報告處理器"""
    
    def __init__(self, api_key: str = None):
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("找不到 API Key！")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """從 PDF 萃取文字"""
        if not PDF_AVAILABLE:
            raise ImportError("需要安裝 pdfplumber: pip3 install pdfplumber")
        
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        return "\n\n".join(text_content)
    
    def structure_report_with_claude(self, report_text: str) -> Dict:
        """使用 Claude 將職能治療報告結構化"""
        
        prompt = f"""請分析以下早期療育（職能治療）評估報告，並將其結構化為 JSON 格式。

報告內容：
{report_text}

請依照以下邏輯進行深度萃取（若無資訊填 null）：

### 1-3. 基本資訊與家屬主訴
- **家屬主訴與期待**：請不要只摘要，請盡量保留「具體描述」。若有提到特定情境（如：學校坐不住、家裡不吃飯），請完整保留。

### 4. 評估結果 (客觀數據與觀察)
重點放在「評估工具的分數」以及「治療師看到的具體行為」。
每個領域請提取：
- domain: 領域 (精細/粗大/感覺/認知/生活自理)
- status: 發展狀態
- assessment_tool: 工具名稱
- quantitative_data: 量化數據 (分數、PR值、SD值)
- qualitative_observation: 質性觀察 (具體的行為描述，如：無法單腳站立超過3秒)

### 5. 問題分析 (核心推理) - 最重要！
這是未來生成報告的關鍵。請將報告中的「問題分析」段落，拆解並整理成以下結構：
- **clinical_reasoning (臨床推理)**: 這是治療師如何將「表現」連結到「根本原因」的過程。
  例如：「寫字字跡潦草（表現）是源於手部肌力不足與本體覺回饋不佳（原因），導致書寫耐力下降（結果）。」
  請嘗試總結報告中的推理邏輯。

### 6. 總結與建議
- 請區分 **strategies_for_home (居家/學校建議)** 與 **therapy_focus (治療課程重點)**。

### 7. 關鍵詞
萃取 8-15 個重要的專業術語，例如：
- 發展相關：精細動作、粗大動作、發展遲緩、發展商數
- 感覺統合：感覺處理、低登錄、警醒度、前庭覺、本體覺
- 認知語言：認知發展、語言理解、指令理解
- 動作技能：手眼協調、工具使用、前三指操作、肌力耐力
- 日常功能：自理能力、遊戲技巧
- 社會情緒：情緒調節、社會互動、分享式注意力

請以以下 JSON 格式回傳（只回傳 JSON，不要其他說明文字）：

{{
  "child_info": {{
    "name_or_id": "兒童姓名或代號",
    "gender": "性別",
    "birth_date": "出生日期（格式：YYYY.MM.DD）",
    "age_at_assessment": "評估時年齡"
  }},
  "assessment_info": {{
    "date": "評估日期",
    "therapist": "治療師姓名",
    "tools": ["評估工具1", "評估工具2"]
  }},
  "family_concerns": [
    "主訴重點1 (例如：在學校無法跟上團體指令)",
    "主訴重點2 (例如：拿筆姿勢不正確)"
  ],
  "assessment_domains": [
    {{
      "domain": "領域名稱",
      "status": "評估狀態",
      "assessment_tool": "評估工具（如有）",
      "observations": "行為觀察與綜合結果",
      "scores": {{
        "description": "分數描述",
        "values": {{
          "百分比": "數值",
          "發展商數": "數值",
          "發展年齡": "數值"
        }}
      }},
      "findings": "主要發現"
    }}
  ],
  "problem_analysis_structured": {{
    "main_issues": ["核心問題1", "核心問題2"],
    "clinical_reasoning_text": "完整的問題分析段落文字（保留原文的語氣與邏輯）",
    "impact_on_function": "這些問題如何影響日常生活（如：影響課堂參與、甚至影響社交互動）"
  }},
  "recommendations": {{
    "treatment_goals": ["治療目標1", "治療目標2"],
    "home_school_strategies": ["給家長/老師的明確建議1", "建議2"],
    "suggested_activities": ["具體訓練活動1", "活動2"]
  }},
  "keywords": ["關鍵詞1", "關鍵詞2"]
}}

請確保：
1. 完整保留報告中的專業術語和數據
2. 評估結果分領域清楚記錄
3. 建議事項按領域分類
4. 關鍵詞涵蓋職能治療的核心概念

"""
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=6000,  # 增加 token 數以處理較長的報告
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            response_text = message.content[0].text
            
            # 移除 markdown 標記
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            structured_data = json.loads(response_text.strip())
            structured_data["processed_at"] = datetime.now().isoformat()
            structured_data["report_type"] = "occupational_therapy"
            
            return structured_data
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤: {e}")
            print(f"Claude 回應前 500 字: {response_text[:500]}")
            return None
        except Exception as e:
            print(f"處理時發生錯誤: {e}")
            return None
    
    def process_single_file(self, file_path: str) -> Dict:
        """處理單一檔案"""
        file_path = Path(file_path)
        
        print(f"處理檔案: {file_path.name}")
        
        # 萃取文字
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(str(file_path))
        elif file_path.suffix.lower() in ['.txt', '.text']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"不支援的檔案格式: {file_path.suffix}")
        
        print(f"  萃取文字長度: {len(text)} 字元")
        
        # 結構化分析
        print(f"  使用 Claude 進行結構化分析...")
        structured_data = self.structure_report_with_claude(text)
        
        if structured_data:
            structured_data["source_file"] = file_path.name
            structured_data["file_path"] = str(file_path.absolute())
            print(f"  ✓ 處理完成")
            
            # 顯示摘要
            print(f"\n  摘要資訊：")
            print(f"    個案: {structured_data.get('child_info', {}).get('name_or_id', 'N/A')}")
            print(f"    年齡: {structured_data.get('child_info', {}).get('age_at_assessment', 'N/A')}")
            print(f"    評估領域數: {len(structured_data.get('assessment_domains', []))}")
            print(f"    關鍵詞數: {len(structured_data.get('keywords', []))}")
        else:
            print(f"  ✗ 處理失敗")
        
        return structured_data



def process_all_raw_files():
    """處理 'raw files' 資料夾中的所有報告"""
    
    print("=" * 70)
    print("大量處理職能治療評估報告")
    print("=" * 70)
    
    # Check for .env and load it
    if DOTENV_AVAILABLE:
        env_path = Path(".") / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
    
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n錯誤：未設定 ANTHROPIC_API_KEY")
        print("請確認 .env 檔案是否設定正確。")
        return
    
    # Initialize processor
    try:
        processor = OccupationalTherapyReportProcessor(api_key=api_key)
    except Exception as e:
        print(f"處理器初始化失敗: {e}")
        return

    # Set up directories
    raw_dir = Path("raw files")
    output_dir = Path("structured files")
    
    if not raw_dir.exists():
        print(f"找不到輸入資料夾: {raw_dir.absolute()}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of files to process
    valid_extensions = ['.pdf', '.txt', '.text']
    files_to_process = [
        f for f in raw_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    
    if not files_to_process:
        print(f"在 {raw_dir} 中找不到可處理的檔案 (.pdf, .txt, .text)")
        return
        
    print(f"找到 {len(files_to_process)} 個檔案待處理...\n")
    
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(files_to_process, 1):
        print(f"[{i}/{len(files_to_process)}] 正在處理: {file_path.name}")
        
        output_filename = f"{file_path.stem}_structured.json"
        output_file = output_dir / output_filename
                
        # Check if output file already exists to avoid redundant processing
        if output_file.exists():
            print(f"   ✓ 檔案已存在，跳過處理: {output_file.name}")
            success_count += 1
            print("-" * 50)
            continue
            
        try:
            result = processor.process_single_file(str(file_path))
            
            if result:
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"   ✓ 已儲存: {output_file.name}")
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            print(f"   ✗ 處理發生例外錯誤: {e}")
            fail_count += 1
            
        print("-" * 50)

    print("\n" + "=" * 70)
    print(f"處理完成！ 成功: {success_count}, 失敗: {fail_count}")
    print(f"輸出目錄: {output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    process_all_raw_files()
