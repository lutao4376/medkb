# -*- coding: utf-8 -*-
"""
问答层：检索 -> 拼材料 -> 让模型只依据材料作答、逐句标出处。

两个刻意的设计：
  * 强制引用：每句事实后面必须有 [n]，回答里出现的每个编号都能在检索结果里找到；
  * 强制弃答：材料里没有就说"未披露"，不许用模型自己的行业知识补——
    作业要看的是"检索到没检索到"，模型瞎补就把这个信号盖掉了。

密钥从文件读，不落到命令行/日志里。
"""
import argparse
import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search import Searcher  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_FILE = r"C:\Users\lt202\Desktop\研一上\人工智能与数据分析\知识库问答 API key.txt"
API = "https://api.deepseek.com/chat/completions"
MODEL = os.environ.get("QA_MODEL", "deepseek-v4-pro")

SYSTEM = """你是医疗器械行业财报分析师。只能依据【材料】回答，不得使用材料以外的知识，也不得自行推算。

规则：
1. 每一处事实、每一个数字后面都要标注来源编号，形如 [3]；一句话里有多个来源就写 [1][4]。
2. 材料里确实没有的，直接回答"财报中未披露该项"，不要猜测、不要用常识补全。
3. 引用金额必须带上单位和报告期（如"2025年度营业收入 332.82 亿元"），
   绝不把不同公司、不同报告期的数字直接放在一起比较。
4. 同一指标有多个口径时（合并/母公司、本期/上期、期末/期初），必须说明引用的是哪一个。
5. 材料是表格时先看清表头：确认这一列是哪一期、什么单位，再引用单元格数字。表头不在本条材料里的，
   就说明"该表格口径需回到原文核对"，不要硬报数字。
6. 语言简洁，能一句说清就不要铺开；不要复述材料原文，直接给结论。"""


def load_key():
    with open(KEY_FILE, "rb") as f:
        return f.read().decode("utf-8-sig").strip()


def build_context(hits):
    parts = []
    for n, c in enumerate(hits, 1):
        sec = c["section"] or "-"
        parts.append(
            f"【材料{n}】{c['name']} {c['year']}年{c['kind']}报告 第{c['page']}页 "
            f"· {sec} · {c['ctype']}\n{c['text']}"
        )
    return "\n\n".join(parts)


def call_llm(messages, key, model=MODEL, temperature=0.0, timeout=180):
    r = requests.post(
        API,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages,
              "temperature": temperature, "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"], j.get("usage", {})


def ask(question, searcher, key=None, topk=8, mode="mix", per_company=3,
        only=None, trace=True, use_llm=True, model=MODEL, kinds="auto", years="auto"):
    hits = searcher.search(question, topk=topk, mode=mode, kinds=kinds,
                           years=years, per_company=per_company, only=only,
                           trace=trace)
    KEEP = ("name", "code", "year", "kind", "section", "page", "seq",
            "ctype", "cite", "_score", "_rank_bm25", "_rank_vec", "_source", "text")
    rec = {"question": question, "mode": mode, "topk": topk,
           "kinds": getattr(searcher, "last_kinds", None),
           "years": getattr(searcher, "last_years", None),
           "panorama": getattr(searcher, "last_panorama", False),
           "n_hits": len(hits),
           "hits": [{k: v for k, v in c.items() if k in KEEP} for c in hits]}

    if not use_llm:
        return {**rec, "answer": None}
    if not hits:
        return {**rec, "answer": "未检索到相关材料。"}

    ctx = build_context(hits)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": f"【材料】\n{ctx}\n\n【问题】\n{question}\n\n"
                    f"请依据以上材料回答，逐处标注编号。"},
    ]
    t0 = time.time()
    answer, usage = call_llm(msgs, key, model=model)
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    rec.update({"answer": answer, "usage": usage,
                "elapsed": round(time.time() - t0, 1),
                "cited": cited,
                "cited_valid": all(1 <= c <= len(hits) for c in cited),
                "n_cited": len(cited)})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--questions", help="批量：一个 json 文件，[{\"q\":...,\"mode\":...}]")
    ap.add_argument("--ids", default=None, help="只跑这些题号，逗号分隔，如 Q1,Q4")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--mode", default="mix",
                    choices=["mix", "bm25", "vec", "panorama"])
    ap.add_argument("--years", default="auto",
                    help="年份过滤：auto(按题面推) / 2025 / all(不限)。只按 --kinds 过滤"
                         "挡不住 2024 年报——它 kind 也是「年度」")
    ap.add_argument("--per-company", type=int, default=3)
    ap.add_argument("--only", default=None, help="只检索这些公司，逗号分隔")
    ap.add_argument("--kinds", default="auto",
                    help="报告期过滤：auto(按题面推) / 年度 / 半年度 / all(不限)")
    ap.add_argument("--no-llm", action="store_true", help="只给检索结果，不调模型")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    s = Searcher()
    print(f"索引：{len(s.chunks)} 块 / {len(s.companies)} 家公司 / "
          f"向量 {'已加载' if s.vectors is not None else '缺失'}", file=sys.stderr)

    only = args.only.split(",") if args.only else None
    kinds = {"all": None, "none": None}.get(args.kinds, args.kinds)
    if kinds not in (None, "auto"):
        kinds = [kinds]
    years = {"all": None, "none": None}.get(args.years, args.years)
    if years not in (None, "auto"):
        years = [years]
    key = None if args.no_llm else load_key()

    if args.questions:
        qs = json.load(open(args.questions, encoding="utf-8"))
        if args.ids:
            want = {x.strip() for x in args.ids.split(",")}
            qs = [it for it in qs if it.get("id") in want]
        results = []
        for i, item in enumerate(qs, 1):
            q = item["q"] if isinstance(item, dict) else item
            # 只放行 ask() 认识的参数。题面里还有 id/type/expected/note，
            # 全塞进去会直接 TypeError。
            kw = {k: v for k, v in (item.items() if isinstance(item, dict) else [])
                  if k in ("mode", "topk", "per_company", "only", "kinds", "years")}
            kw.setdefault("mode", args.mode)
            kw.setdefault("topk", args.topk)
            kw.setdefault("only", only)
            kw.setdefault("years", years)
            print(f"[{i}/{len(qs)}] {item.get('id','?')} {q[:60]}", file=sys.stderr)
            r = ask(q, s, key, use_llm=not args.no_llm, model=args.model, **kw)
            r["_id"] = item.get("id") if isinstance(item, dict) else None
            r["_type"] = item.get("type") if isinstance(item, dict) else None
            r["_expected"] = item.get("expected") if isinstance(item, dict) else None
            r["_note"] = item.get("note") if isinstance(item, dict) else None
            results.append(r)
        out = args.out or os.path.join(ROOT, "out", "qa_results.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(results, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n{len(results)} 题 -> {out}")
        return

    if not args.question:
        ap.error("给个问题，或用 --questions 批量")
    r = ask(args.question, s, key, topk=args.topk, mode=args.mode,
            per_company=args.per_company, only=only, kinds=kinds, years=years,
            use_llm=not args.no_llm, model=args.model)

    print("=" * 70)
    print(f"问题：{r['question']}")
    print("=" * 70)
    print(r["answer"] or "(未调模型)")
    print("\n" + "-" * 70)
    print(f"召回 {r['n_hits']} 块：")
    for n, h in enumerate(r["hits"], 1):
        tag = f"{h['_source']:4}" if "_source" in h else ""
        print(f"  [{n}] {tag} {h['cite']} · {h['section'][:16]:16} "
              f"bm25#{h.get('_rank_bm25') or '-'} vec#{h.get('_rank_vec') or '-'}")
    if r.get("cited"):
        print(f"\n引用编号 {r['cited']}  合法={r['cited_valid']}  "
              f"用时{r['elapsed']}s  tokens={r.get('usage',{}).get('total_tokens')}")


if __name__ == "__main__":
    main()
