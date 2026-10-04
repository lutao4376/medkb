# -*- coding: utf-8 -*-
"""下载 Qwen3-Embedding-0.6B 到 D 盘。huggingface.co 直连不通，走 hf-mirror。"""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\hf-cache")
# huggingface_hub 新版默认走 Xet 存储，会去连 cas-server.xethub.hf.co，
# 而镜像站不代理这个域名 -> 401 重建失败。关掉它，退回普通 HTTP 下载。
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.stdout.reconfigure(encoding="utf-8")

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def main():
    from huggingface_hub import snapshot_download

    print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
    print(f"HF_HOME     = {os.environ['HF_HOME']}")
    t0 = time.time()
    path = snapshot_download(
        MODEL,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.py"],
        max_workers=4,
    )
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(path) for f in fs)
    print(f"\n已下载到: {path}")
    print(f"体积 {total/1e9:.2f} GB，用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
