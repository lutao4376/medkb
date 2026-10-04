# -*- coding: utf-8 -*-
"""
给文本块建两套索引：BM25（jieba 分词）+ 向量（Qwen3-Embedding-0.6B）。
两套索引召回的结果用 RRF 融合——两边的分数量纲没法比，融合名次最稳。

产物都在 data/index/：
  chunks.jsonl      原样拷一份，索引和块一一对应
  bm25.pkl          倒排 + idf
  vectors.npy       N×1024 float32，已 L2 归一化
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter

import jieba
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
IDX = os.path.join(DATA, "index")
CHUNKS = os.path.join(DATA, "chunks.jsonl")
MODEL = "Qwen/Qwen3-Embedding-0.6B"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"

PUNCT = re.compile(r"^[\s\W_]+$", re.U)


def tokenize(s):
    """jieba 搜索引擎模式：粒度细，召回高。滤掉纯标点。"""
    return [t for t in jieba.lcut_for_search(s) if t.strip() and not PUNCT.match(t)]


# ---------------------------------------------------------------- BM25
class BM25:
    """
    Okapi BM25。自己写是为了少装一个包（网络太慢）。
    倒排表 postings[term] = [(doc_i, tf), ...]，打分只碰命中的文档。
    """

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0
        postings = {}
        for i, d in enumerate(docs_tokens):
            for w, c in Counter(d).items():
                postings.setdefault(w, []).append((i, c))
        self.postings = postings
        self.idf = {
            w: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
            for w, p in postings.items()
        }

    def search(self, q_tokens, topk=50):
        s = np.zeros(self.N, dtype=np.float32)
        for w in set(q_tokens):
            p = self.postings.get(w)
            if not p:
                continue
            idf = self.idf[w]
            for i, f in p:
                s[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                )
        n = min(topk, self.N)
        idx = np.argpartition(-s, n - 1)[:n]
        idx = idx[np.argsort(-s[idx])]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]


# ---------------------------------------------------------------- 建索引
# BM25 索引哪个字段。测过两种（src/bench_bm25_field.py）：
#   embed_text = 公司名+章节+页码+正文；每块都带公司全称，样板词把 idf 拉平，
#                封面页/签字页反而排第一（"迈瑞2025营业收入"召回的是2024年报封面）。
#   text       = 纯正文；正确年份的块能排到第一。
# 所以 BM25 吃正文。向量那边仍用 embed_text——表头对语义匹配是有效信息，不冲突。
BM25_FIELD = "text"


def build_bm25(chunks, field=BM25_FIELD):
    print(f"BM25: jieba 分词中（字段 {field}）…")
    t0 = time.time()
    toks = []
    for i, c in enumerate(chunks, 1):
        toks.append(tokenize(c[field]))
        if i % 5000 == 0:
            print(f"  {i}/{len(chunks)}")
    print(f"  分词完 {time.time()-t0:.0f}s，建倒排 …")
    bm = BM25(toks)
    print(f"  词表 {len(bm.postings)}，均长 {bm.avgdl:.0f} 词/块")
    return bm


SHARD = 4000      # 每 4000 块落一次盘。三个半小时的任务，崩了不能从头再来


def build_vectors(chunks, batch=None, limit=None, device=None, shard=SHARD,
                  out_dir=None, log=None):
    """
    分片编码 + 断点续跑：每片单独存 vec_shard_XXXX.npy，重跑时已存在的片直接跳过。
    片内顺序 = chunks 顺序，最后按片号拼接即可。
    """
    from encoder import Encoder
    out_dir = out_dir or IDX
    os.makedirs(out_dir, exist_ok=True)
    say = lambda *a: print(*a, **(dict(file=log, flush=True) if log else {}))

    texts = [c["embed_text"] for c in chunks]
    if limit:
        texts = texts[:limit]
    n = len(texts)
    n_shard = (n + shard - 1) // shard

    todo = [i for i in range(n_shard)
            if not os.path.exists(os.path.join(out_dir, f"vec_shard_{i:04d}.npy"))]
    if not todo:
        say(f"  全部 {n_shard} 片已存在，直接拼接")
    else:
        e = Encoder(device=device)
        say(f"  设备 {e.device}  维度 {e.dim}  共 {n} 块 / {n_shard} 片，"
            f"需跑 {len(todo)} 片")
        # 先统计全库 token 数。按块数估 ETA 会偏乐观（各片块长差一倍），按 token 才算得准。
        say("  统计各块 token 数（估时用）…")
        all_lens = [len(e.tk(t, truncation=True, max_length=1536)["input_ids"]) for t in texts]
        total_tok = sum(all_lens)
        say(f"  共 {total_tok/1e6:.1f}M token，均值 {total_tok/n:.0f}/块")

        t0 = time.time()
        already = sum(min(shard, n - i * shard) for i in range(n_shard)
                      if i not in todo)          # 历史片已含的块数
        done = already
        tok_done = sum(all_lens[:already if already else 0])
        prog = os.path.join(out_dir, "progress.json")

        def write_prog(st, **kw):
            """给进度条读的状态文件。每批覆盖一次，键名保持稳定。"""
            d = {"shard": st["shard"], "n_shard": n_shard, "done": st["done"],
                 "total": n, "elapsed": st["elapsed"], "ts": time.time(),
                 "waste": st.get("waste", 0.0),
                 "done_tok": st.get("done_tok", 0), "total_tok": total_tok}
            run = st["done"] - already             # 本次真正跑出来的块数
            rate = run / st["elapsed"] if st["elapsed"] > 0 else 0.0
            d["rate_item"] = rate
            d["eta_item"] = (n - st["done"]) / rate if rate > 0 else 0.0
            tdone = st.get("done_tok", 0)
            trate = (tdone - tok_done) / st["elapsed"] if st["elapsed"] > 0 else 0.0
            d["rate_tok"] = trate
            d["eta"] = (total_tok - tdone) / trate if trate > 0 else d["eta_item"]
            d.update(kw)
            try:
                tmp = prog + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False)
                os.replace(tmp, prog)
            except Exception:
                pass

        for k, si in enumerate(todo, 1):
            s, en = si * shard, min((si + 1) * shard, n)
            base, tok_base = done, tok_done        # 本片开始前的进度

            def hook(di, dn, dt, dtn, el, _si=si, _b=base, _tb=tok_base):
                write_prog({"shard": _si, "done": _b + di, "elapsed": el,
                            "done_tok": _tb + dt})

            v = e.encode(texts[s:en], on_batch=hook)
            np.save(os.path.join(out_dir, f"vec_shard_{si:04d}.npy"), v)
            done += en - s
            tok_done += sum(all_lens[s:en])
            el = time.time() - t0
            write_prog({"shard": si, "done": done, "elapsed": el,
                        "done_tok": tok_done, "waste": e.stats["waste"]})
            rate = (done - already) / el if el else 0      # 速度只按本次跑的算
            eta = (total_tok - tok_done) / (tok_done - 0) * el if tok_done else 0
            say(f"  片 {si+1}/{n_shard} 存盘  {done}/{n} 块  "
                f"{rate:.1f}块/s  {(tok_done-tok_base)/max(1e-9,(time.time()-t0)):.0f}tok/s  "
                f"已用{el/60:.1f}分"
                f"  (padding 浪费 {e.stats['waste']*100:.0f}%)")

    parts = [np.load(os.path.join(out_dir, f"vec_shard_{i:04d}.npy"))
             for i in range(n_shard)]
    out = np.concatenate(parts, axis=0)
    say(f"  拼接完成 {out.shape}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bm25", action="store_true")
    ap.add_argument("--skip-vec", action="store_true")
    ap.add_argument("--bm25-field", default=BM25_FIELD,
                    choices=["text", "embed_text"],
                    help="BM25 索引哪个字段，默认 text（见 BM25_FIELD 注释）")
    ap.add_argument("--limit", type=int, default=None, help="只编码前 N 块，用于估时")
    args = ap.parse_args()

    os.makedirs(IDX, exist_ok=True)
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    print(f"读入 {len(chunks)} 块")
    if args.limit:
        chunks = chunks[: args.limit]
        print(f"--limit 生效，只用前 {len(chunks)} 块（估时用，不落盘）")

    if not args.skip_bm25 and not args.limit:
        bm = build_bm25(chunks, field=args.bm25_field)
        p = os.path.join(IDX, f"bm25_{args.bm25_field}.pkl")
        with open(p, "wb") as f:
            pickle.dump({"postings": bm.postings, "idf": bm.idf, "dl": bm.dl,
                         "avgdl": bm.avgdl, "k1": bm.k1, "b": bm.b, "N": bm.N,
                         "field": args.bm25_field}, f)
        print(f"  -> {p}  {os.path.getsize(p)/1e6:.1f}MB")

    if not args.skip_vec:
        t0 = time.time()
        vecs = build_vectors(chunks, limit=args.limit)
        if not args.limit:
            np.save(os.path.join(IDX, "vectors.npy"), vecs)
            json.dump([c["code"] + "|" + str(c["page"]) + "|" + str(c["seq"])
                       for c in chunks],
                      open(os.path.join(IDX, "chunk_ids.json"), "w", encoding="utf-8"))
            with open(os.path.join(IDX, "chunks.jsonl"), "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"  -> vectors.npy {vecs.nbytes/1e6:.0f}MB，用时 {(time.time()-t0)/60:.1f} 分")
        else:
            print(f"  估时模式，未落盘")


if __name__ == "__main__":
    main()
