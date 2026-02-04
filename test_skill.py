#!/usr/bin/env python3
"""
測試 OT Report Generation Skill

這個腳本用於測試 skill 的 prompt 模組是否正常運作。
"""

import sys
import os

# 加入 skill 路徑
SKILL_PATH = os.path.join(os.path.dirname(__file__), '.agent', 'skills', 'ot-report-generation')
sys.path.insert(0, SKILL_PATH)

def test_import():
    """測試模組導入"""
    print("=" * 60)
    print("測試 1: 模組導入")
    print("=" * 60)
    
    try:
        from prompts import get_system_prompt, get_user_prompt, get_prompt_metadata
        print("✅ 成功導入 prompt 模組")
        return True
    except ImportError as e:
        print(f"❌ 導入失敗: {e}")
        return False

def test_metadata():
    """測試元資料"""
    print("\n" + "=" * 60)
    print("測試 2: Prompt 元資料")
    print("=" * 60)
    
    try:
        from prompts import get_prompt_metadata
        metadata = get_prompt_metadata()
        
        print(f"版本: {metadata['version']}")
        print(f"名稱: {metadata['name']}")
        print(f"描述: {metadata['description']}")
        print(f"建立日期: {metadata['created_date']}")
        print(f"輸出區塊: {', '.join(metadata['output_sections'])}")
        print("✅ 元資料正常")
        return True
    except Exception as e:
        print(f"❌ 元資料測試失敗: {e}")
        return False

def test_system_prompt():
    """測試系統 prompt"""
    print("\n" + "=" * 60)
    print("測試 3: 系統 Prompt")
    print("=" * 60)
    
    try:
        from prompts import get_system_prompt
        system_prompt = get_system_prompt()
        
        # 檢查是否包含關鍵字
        keywords = ["職能治療師", "台灣繁體中文", "專業術語", "臨床推理"]
        missing_keywords = [kw for kw in keywords if kw not in system_prompt]
        
        if missing_keywords:
            print(f"⚠️  警告: 以下關鍵字未出現在 system prompt 中: {missing_keywords}")
        else:
            print("✅ System prompt 包含所有必要關鍵字")
        
        print(f"\nPrompt 長度: {len(system_prompt)} 字元")
        print(f"前 100 字元: {system_prompt[:100]}...")
        return True
    except Exception as e:
        print(f"❌ System prompt 測試失敗: {e}")
        return False

def test_user_prompt():
    """測試使用者 prompt"""
    print("\n" + "=" * 60)
    print("測試 4: 使用者 Prompt")
    print("=" * 60)
    
    try:
        from prompts import get_user_prompt
        
        # 模擬資料
        context_str = "【參考案例】精細動作：工具使用經驗不足..."
        case_description = "孩子寫字很醜，抓握姿勢不成熟"
        
        user_prompt = get_user_prompt(context_str, case_description)
        
        # 檢查是否正確插入資料
        if context_str in user_prompt and case_description in user_prompt:
            print("✅ User prompt 正確組合參考案例與個案描述")
        else:
            print("❌ User prompt 未正確組合資料")
            return False
        
        # 檢查格式要求
        format_keywords = ["問題分析", "總結與建議", "格式規範", "階層規範"]
        missing_format = [kw for kw in format_keywords if kw not in user_prompt]
        
        if missing_format:
            print(f"⚠️  警告: 以下格式要求未出現: {missing_format}")
        else:
            print("✅ User prompt 包含所有格式要求")
        
        print(f"\nPrompt 長度: {len(user_prompt)} 字元")
        return True
    except Exception as e:
        print(f"❌ User prompt 測試失敗: {e}")
        return False

def test_integration():
    """整合測試"""
    print("\n" + "=" * 60)
    print("測試 5: 整合測試（模擬實際使用）")
    print("=" * 60)
    
    try:
        from prompts import get_system_prompt, get_user_prompt
        
        # 模擬實際使用情境
        context_str = """
【針對「精細動作」的歷史參考資料 (0.85)】
核心問題：
1. 精細動作：工具使用經驗不足，運筆技巧處初階階段

總結與建議：
1. 綜合以上結果，建議安排職能療育課程。
2. 精細動作：透過操作不同粗細的工具練習握筆...
"""
        
        case_description = """
家屬表示孩子寫字很醜，常常握筆姿勢不正確。
觀察發現：
1. 精細動作：使用全手掌握筆，運筆力道不穩定
2. 手眼協調：描寫線條時常常超出範圍
"""
        
        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(context_str, case_description)
        
        # 模擬 LLM 訊息格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print("✅ 成功組合 LLM 訊息格式")
        print(f"\n訊息數量: {len(messages)}")
        print(f"System prompt 長度: {len(messages[0]['content'])} 字元")
        print(f"User prompt 長度: {len(messages[1]['content'])} 字元")
        
        return True
    except Exception as e:
        print(f"❌ 整合測試失敗: {e}")
        return False

def main():
    """執行所有測試"""
    print("\n🧪 開始測試 OT Report Generation Skill\n")
    
    tests = [
        test_import,
        test_metadata,
        test_system_prompt,
        test_user_prompt,
        test_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # 總結
    print("\n" + "=" * 60)
    print("測試總結")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通過: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有測試通過！Skill 已準備就緒。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 個測試失敗，請檢查錯誤訊息。")
        return 1

if __name__ == "__main__":
    exit(main())
