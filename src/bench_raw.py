# -*- coding: utf-8 -*-
import os,sys,time
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoTokenizer, AutoModel

name="Qwen/Qwen3-Embedding-0.6B"
tk=AutoTokenizer.from_pretrained(name, padding_side="left")
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("线程数:", torch.get_num_threads(), "| cudnn:", torch.backends.cudnn.version())

def run(dev, bs, sl, dtype, n=3):
    m=AutoModel.from_pretrained(name, dtype=dtype).to(dev).eval()
    ids=torch.randint(1000,15000,(bs,sl))
    am=torch.ones_like(ids)
    with torch.inference_mode():
        for _ in range(2):
            out=m(input_ids=ids.to(dev),attention_mask=am.to(dev)).last_hidden_state
        if dev=="cuda": torch.cuda.synchronize()
        t0=time.time()
        for _ in range(n):
            out=m(input_ids=ids.to(dev),attention_mask=am.to(dev)).last_hidden_state
        if dev=="cuda": torch.cuda.synchronize()
        el=(time.time()-t0)/n
    print(f"  {dev:4} {str(dtype):15} bs={bs} sl={sl}: {el*1000:7.1f} ms/次  "
          f"{bs*sl/el:8.0f} tok/s  ({bs/el:.1f} 条/s)")
    del m, out
    if dev=="cuda": torch.cuda.empty_cache()

run("cuda", 64, 512, torch.float16)
run("cuda", 64, 512, torch.bfloat16)
run("cuda", 64, 512, torch.float32)
run("cuda", 64, 1536, torch.float16)
