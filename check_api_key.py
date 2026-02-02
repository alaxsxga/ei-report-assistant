#!/usr/bin/env python3
"""
API Key 設定檢查工具
檢查 .env 檔案和 API Key 是否正確設定
"""

import os
from pathlib import Path

def check_env_file():
    """檢查 .env 檔案"""
    print("=" * 70)
    print("API Key 設定檢查")
    print("=" * 70)
    
    # 檢查 .env 檔案
    env_file = Path(".env")
    
    print("\n步驟 1: 檢查 .env 檔案")
    print("-" * 70)
    
    if not env_file.exists():
        print("❌ 找不到 .env 檔案")
        print("\n請執行以下指令建立 .env 檔案：")
        print("echo 'ANTHROPIC_API_KEY=sk-ant-api03-你的金鑰' > .env")
        return False
    else:
        print("✅ .env 檔案存在")
    
    # 讀取內容
    print("\n步驟 2: 檢查 .env 內容")
    print("-" * 70)
    
    with open(env_file, 'r') as f:
        content = f.read().strip()
    
    if not content:
        print("❌ .env 檔案是空的")
        print("\n請在 .env 中加入：")
        print("ANTHROPIC_API_KEY=sk-ant-api03-你的金鑰")
        return False
    
    # 檢查格式
    if "ANTHROPIC_API_KEY=" not in content:
        print("❌ .env 格式不正確")
        print(f"\n目前內容：{content[:50]}")
        print("\n正確格式應該是：")
        print("ANTHROPIC_API_KEY=sk-ant-api03-你的金鑰")
        return False
    
    # 萃取 API Key
    api_key = content.split("ANTHROPIC_API_KEY=")[1].split("\n")[0].strip()
    
    if not api_key:
        print("❌ API Key 是空的")
        return False
    
    if not api_key.startswith("sk-ant-"):
        print("⚠️  API Key 格式可能不正確")
        print(f"   API Key 應該以 'sk-ant-' 開頭")
        print(f"   目前的值: {api_key[:20]}...")
    else:
        print("✅ .env 格式正確")
        print(f"   API Key: {api_key[:20]}...{api_key[-4:]}")
    
    # 檢查環境變數
    print("\n步驟 3: 檢查是否能讀取 API Key")
    print("-" * 70)
    
    # 嘗試用 dotenv 載入
    try:
        from dotenv import load_dotenv
        load_dotenv()
        env_api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if env_api_key:
            print("✅ 可以成功讀取 API Key")
            print(f"   讀取到的值: {env_api_key[:20]}...{env_api_key[-4:]}")
        else:
            print("❌ 無法讀取 API Key")
            return False
            
    except ImportError:
        print("⚠️  python-dotenv 未安裝")
        print("   請執行: pip3 install python-dotenv")
        print("\n   不過你也可以用環境變數方式：")
        print(f"   export ANTHROPIC_API_KEY='{api_key}'")
        return False
    
    # 測試 API
    print("\n步驟 4: 測試 API 連線（選配）")
    print("-" * 70)
    print("提示：這需要網路連線和 anthropic 套件")
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=env_api_key)
        
        # 簡單測試
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi"}]
        )
        
        print("✅ API 連線成功！")
        print(f"   Claude 回應: {response.content[0].text[:50]}")
        
    except ImportError:
        print("⚠️  anthropic 套件未安裝")
        print("   請執行: pip3 install anthropic")
    except Exception as e:
        print(f"⚠️  API 測試失敗: {str(e)[:100]}")
        print("   這可能是因為：")
        print("   1. API Key 不正確")
        print("   2. 沒有網路連線")
        print("   3. API 額度不足")
    
    # 總結
    print("\n" + "=" * 70)
    print("檢查完成！")
    print("=" * 70)
    print("\n下一步：")
    print("如果所有檢查都通過，可以執行：")
    print("  python3 test_real_report.py")
    print()
    
    return True


if __name__ == "__main__":
    check_env_file()
