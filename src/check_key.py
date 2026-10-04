# -*- coding: utf-8 -*-
"""检查知识库问答 API key 是否可用。只打印状态与模型名，不打印密钥本身。"""
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")

KEY_FILE = r"C:\Users\lt202\Desktop\研一上\人工智能与数据分析\知识库问答 API key.txt"


def load_key():
    with open(KEY_FILE, "rb") as f:
        return f.read().decode("utf-8-sig").strip()


def main():
    key = load_key()
    print(f"读取成功: 长度={len(key)}  前3位={key[:3]!r}  无空白={' ' not in key}")

    for name, url in [
        ("DeepSeek", "https://api.deepseek.com/models"),
    ]:
        try:
            r = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=25)
            print(f"{name}: HTTP {r.status_code}")
            if r.ok:
                print("  可用模型:", [m.get("id") for m in r.json().get("data", [])])
            else:
                print("  ", r.text[:200])
        except Exception as e:
            print(f"{name}: ERR {e}")


if __name__ == "__main__":
    main()
