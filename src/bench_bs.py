import sys,time
import os
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ["HF_HUB_OFFLINE"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoModel
m=AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.float16).to("cuda").eval()
for bs in (32,64,128,256):
    for sl in (512,768):
        try:
            ids=torch.randint(1000,15000,(bs,sl),device="cuda"); am=torch.ones_like(ids)
            with torch.inference_mode():
                for _ in range(2): m(input_ids=ids,attention_mask=am)
                torch.cuda.synchronize(); t0=time.time()
                for _ in range(3): m(input_ids=ids,attention_mask=am)
                torch.cuda.synchronize(); el=(time.time()-t0)/3
            fl=2*0.6e9*bs*sl
            print(f"  bs={bs:4} sl={sl}: {el*1000:7.1f} ms  {bs*sl/el:7.0f} tok/s  {fl/el/1e12:5.1f} TFLOPS")
            del ids,am
        except RuntimeError as e:
            print(f"  bs={bs:4} sl={sl}: OOM")
        torch.cuda.empty_cache()
