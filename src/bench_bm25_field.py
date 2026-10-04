# -*- coding: utf-8 -*-
"""
BM25 索引字段 A/B：用 embed_text（公司名+章节+页码+正文）还是只用 text（正文）。

动机：embed_text 里每块都带公司全称，样板文字把 idf 拉平，封面页/签字页反而
排到最前。这里建一份只吃正文的 BM25，拿几道真题并排看。

产物：data/index/bm25_text.pkl（若选用正文口径，改 index.py 的字段即可）
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
from index import BM25, tokenize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "data", "index")

QS = [
    "迈瑞医疗2025年的营业收入是多少",
    "安图生物的体外诊断试剂毛利率是多少",
    "哪些公司在年报里提到了带量采购",
    "各家公司2025年的研发费用分别是多少",
    "联影医疗的研发投入占营业收入比例",
]


def main():
    chunks = [json.loads(l) for l in
              open(os.path.join(IDX, "chunks.jsonl"), encoding="utf-8")]
    print(f"{len(chunks)} 块")

    t0 = time.time()
    toks = []
    for c in chunks:
        toks.append(tokenize(c["text"]))          # 只看正文
    print(f"分词 {time.time()-t0:.0f}s")

    bm = BM25(toks)
    print(f"词表 {len(bm.postings)}  均长 {bm.avgdl:.0f} 词/块")

    with open(os.path.join(IDX, "bm25_text.pkl"), "wb") as f:
        pickle.dump({"postings": bm.postings, "idf": bm.idf, "dl": bm.dl,
                     "avgdl": bm.avgdl, "k1": bm.k1, "b": bm.b,
                     "N": bm.N}, f)

    # 旧口径（embed_text）读回来对比
    with open(os.path.join(IDX, "bm25.pkl"), "rb") as f:
        d = pickle.load(f)
    old = BM25([[]])
    for k in ("postings", "idf", "dl", "avgdl", "k1", "b", "N"):
        setattr(old, k, d[k])

    for q in QS:
        qt = tokenize(q)
        print("\n" + "=" * 96)
        print(f"Q: {q}")
        for tag, b in (("embed_text", old), ("text", bm)):
            print(f"  --- {tag} ---")
            for r, (i, sc) in enumerate(b.search(qt, topk=5), 1):
                c = chunks[i]
                t = c["text"].replace("\n", " ")[:56]
                print(f"   {r}. {sc:6.3f} {c['name']:5} {c['year']}{c['kind'][:2]}"
                      f" p{c['page']:<4} {t}")
    print(f"\n-> {os.path.join(IDX,'bm25_text.pkl')}")


if __name__ == "__main__":
    main()
