# -*- coding: utf-8 -*-
import os, sys, json, time, random
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ["HF_HUB_DISABLE_XET"]="1"
sys.stdout.reconfigure(encoding="utf-8")
import torch, numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

cs=[json.loads(l) for l in open("data/chunks.jsonl",encoding="utf-8")]
random.seed(1); samp=random.sample(cs, 192)
texts=[c["embed_text"] for c in samp]

tk=AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
print("分词器类型:", type(tk).__name__, "| is_fast:", tk.is_fast)

t0=time.time(); enc=tk(texts[:64], padding=True, truncation=True, max_length=1536)
el=time.time()-t0
ntok=sum(len(x) for x in enc["input_ids"])
print(f"CPU 分词 64 条(含padding {ntok} tok): {el:.2f}s = {ntok/el:.0f} tok/s  ")

m=SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
m.max_seq_length=1536
print("attention 实现:", m[0].auto_model.config._attn_implementation)

for tag, half, bs, ml in [("fp16/1536/bs64",True,64,1536),
                          ("fp32/1536/bs32",False,32,1536),
                          ("fp16/512/bs64",True,64,512)]:
    m.max_seq_length=ml
    if half: m.half()
    else: m.float()
    sub=[t[:ml*2] for t in texts[:bs]]
    m.encode(sub, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    torch.cuda.synchronize(); t0=time.time()
    m.encode(sub, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    torch.cuda.synchronize(); el=time.time()-t0
    n=sum(len(x) for x in tk(sub, truncation=True, max_length=ml)["input_ids"])
    print(f"  {tag}: {el:.2f}s / {bs}条 = {bs/el:.1f}块/s , {n/el:.0f} tok/s")
