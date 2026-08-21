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
        from prompts import (
            get_json_system_prompt, get_json_user_prompt,
            get_segmentation_system_prompt, get_segmentation_user_prompt,
            get_prompt_metadata
        )
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

        print(f"名稱: {metadata['name']}")
        print(f"描述: {metadata['description']}")
        print(f"語言: {metadata['language']}")
        print(f"輸出區塊: {', '.join(metadata['output_sections'])}")
        print("✅ 元資料正常")
        return True
    except Exception as e:
        print(f"❌ 元資料測試失敗: {e}")
        return False

def test_segmentation_prompt():
    """測試區塊拆解 prompt"""
    print("\n" + "=" * 60)
    print("測試 3: 區塊拆解 Prompt")
    print("=" * 60)

    try:
        from prompts import get_segmentation_system_prompt, get_segmentation_user_prompt

        known_domains = {"精細動作", "感覺統合", "認知發展"}
        case_description = "精細動作：慣用手尚未穩定，操作經驗不足，抓握能力偏弱。"

        system_prompt = get_segmentation_system_prompt()
        user_prompt = get_segmentation_user_prompt(case_description, known_domains)

        if case_description not in user_prompt:
            print("❌ user prompt 未正確包含原文")
            return False
        if not all(d in user_prompt for d in known_domains):
            print("❌ user prompt 未正確包含已知領域清單")
            return False

        print(f"System prompt 長度: {len(system_prompt)} 字元")
        print(f"User prompt 長度: {len(user_prompt)} 字元")
        print("✅ 區塊拆解 prompt 正確組合")
        return True
    except Exception as e:
        print(f"❌ 區塊拆解 prompt 測試失敗: {e}")
        return False

def test_json_prompt():
    """測試結構化生成 prompt"""
    print("\n" + "=" * 60)
    print("測試 4: 結構化生成 Prompt")
    print("=" * 60)

    try:
        from prompts import get_json_system_prompt, get_json_user_prompt

        domain_blocks = [
            {
                "domain": "精細動作",
                "case_issue": "抓握姿勢不成熟，運筆力道不穩定",
                "reference": "【針對「精細動作」的歷史參考資料】工具使用經驗不足，運筆技巧處初階階段"
            }
        ]

        system_prompt = get_json_system_prompt()
        user_prompt = get_json_user_prompt(domain_blocks)

        if "精細動作" not in user_prompt or domain_blocks[0]["case_issue"] not in user_prompt:
            print("❌ user prompt 未正確組合 domain_blocks 內容")
            return False

        format_keywords = ["issue_summary", "recommendation", "course_recommendation"]
        missing_format = [kw for kw in format_keywords if kw not in user_prompt]
        if missing_format:
            print(f"⚠️  警告: 以下欄位要求未出現: {missing_format}")
        else:
            print("✅ User prompt 包含所有必要欄位")

        print(f"System prompt 長度: {len(system_prompt)} 字元")
        print(f"User prompt 長度: {len(user_prompt)} 字元")
        return True
    except Exception as e:
        print(f"❌ 結構化生成 prompt 測試失敗: {e}")
        return False

def main():
    """執行所有測試"""
    print("\n🧪 開始測試 OT Report Generation Skill\n")

    tests = [
        test_import,
        test_metadata,
        test_segmentation_prompt,
        test_json_prompt
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
