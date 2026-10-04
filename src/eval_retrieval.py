# -*- coding: utf-8 -*-
"""
纯检索评测：不调模型，只问"该找到的数字，在被召回的块里找得到吗"。

为什么要这一层：LLM 一次回答 35 秒，10 题就是 6 分钟，改一个检索参数要重跑一遍
根本没法调。而"检索有没有覆盖到"跟模型无关，可以秒级测出来。模型答错有一大半是
因为材料根本没召回，先把这个信号量出来。

指标：
  数值覆盖 = 要求的原始披露值里，有多少能在 top-k 块里找到
  报告期纯度 = 召回块里属于期望报告期（默认"年度"）的比例

用法：
    python src/eval_retrieval.py --modes bm25 bm25 --kinds auto none
    python src/eval_retrieval.py --modes bm25 vec mix --kinds auto
"""
import argparse
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "data", "index")


def norm(s):
    """去掉逗号和所有空白，只为了在里面找长数字串。"""
    return re.sub(r"[\s,，]", "", s)


def forms(v):
    """一个数值可能的各种写法。"""
    if v is None:
        return []
    out = {f"{v:.2f}", f"{v:,.2f}", f"{v:.4f}"}
    if abs(v - round(v)) < 1e-9:
        out.add(str(int(round(v))))
    return {norm(x) for x in out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["bm25"],
                    help="bm25 / vec / mix / panorama（panorama = 逐家扇出的 mix）")
    ap.add_argument("--kinds", nargs="+", default=["auto"],
                    help="auto(按题面推) / none(不限) / 年度 / 半年度")
    ap.add_argument("--years", nargs="+", default=["auto"],
                    help="auto(按题面推) / none(不限) / 2025。"
                         "kinds 只挡得住半年报，挡不住 2024 年报——它 kind 也是年度")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--per-company", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    key = json.load(open(os.path.join(ROOT, "data", "answer_key.json"), encoding="utf-8"))
    qs = json.load(open(os.path.join(ROOT, "data", "questions.json"), encoding="utf-8"))
    qmap = {q["id"]: q for q in qs}

    order = sorted(key, key=lambda n: -key[n]["revenue_2025"])
    TOP6, ALL = order[:6], order

    # 每题要求被召回覆盖的原始值
    REQ = {
        "Q1": [("迈瑞医疗", ["revenue_2025", "np_2025", "dnp_2025"])],
        "Q2": [("联影医疗", ["revenue_2025", "revenue_2024", "np_2025", "np_2024"])],
        "Q3": [(n, ["revenue_2025"]) for n in TOP6],
        "Q4": [(n, ["revenue_2025", "np_2025"]) for n in ALL],
        "Q5": [(n, ["revenue_2025", "revenue_2024"]) for n in ALL],
        "Q8": [(n, ["np_2025", "np_2024"]) for n in ALL],
        "Q9": [("大博医疗", ["np_2025", "dnp_2025"])],
    }
    # 只存在于正文段落里、不在主要会计数据表中的数字
    EXTRA = {
        "Q7": [("大博医疗", [71130088.33, 311984374.13])],   # 齿科类 / 境外业务
    }

    from search import Searcher, infer_kinds   # noqa: E402
    s = Searcher()
    print(f"索引 {len(s.chunks)} 块；向量 {'已加载' if s.vectors is not None else '缺失'}\n")

    results = {}
    # mode × kinds × years 的组合，拍平成一个列表再循环：三层嵌套的话，最里层
    # 那两个 4 空格缩进进去，条件行就没法看了
    plans = [(m, k, y) for m in args.modes for k in args.kinds for y in args.years]
    for mode, kind, yr in plans:
        if mode == "vec" and s.vectors is None:
            continue
        tag = f"{mode}/{kind}" + ("" if yr == "auto" else f"/年={yr}")
        rows = []
        for q in qs:
            qid, text = q["id"], q["q"]
            kk = None if kind == "none" else kind
            yy = None if yr == "none" else yr
            hits = s.search(text, topk=args.topk, mode=mode,
                            per_company=args.per_company, kinds=kk,
                            years=yy, trace=True)
            blob = norm("\n".join(h["text"] for h in hits))
            exp_kind = (infer_kinds(text) or [None])[0]
            purity = (sum(1 for h in hits if exp_kind is None or h["kind"] == exp_kind)
                      / max(1, len(hits)))
            reps = {(h["name"], h["year"], h["kind"]) for h in hits}
            vals = []
            for co, fields in REQ.get(qid, []):
                vals += [(f"{co}.{f}", key[co]["raw"].get(f)) for f in fields]
            vals += [(f"{co}.{v}", v) for co, vs in EXTRA.get(qid, []) for v in vs]
            if vals:
                need = tot = 0
                miss = []
                for label, v in vals:
                    if v is None:
                        continue
                    tot += 1
                    if any(x in blob for x in forms(v) if len(x) > 5):
                        need += 1
                    else:
                        miss.append(label)
                cov = f"{need}/{tot}"
            else:
                cov, miss = "—", []
            rows.append((qid, len(hits), len(reps), purity, cov, miss))
        results[tag] = {"rows": rows, "qmap": qmap}

    # 打印
    for tag, d in results.items():
        print("=" * 96)
        print(f"■ {tag}")
        print(f"  {'题':4} {'块':>3} {'报告':>4} {'报告期纯度':>10} {'数值覆盖':>9}  缺")
        for qid, nh, nr, pu, cov, miss in d["rows"]:
            m = ("、".join(miss[:4]) + ("…" if len(miss) > 4 else "")) if miss else ""
            print(f"  {qid:4} {nh:3} {nr:4} {pu*100:9.0f}% {cov:>9}  {m}")
        tots = [r[4] for r in d["rows"] if r[4] != "—"]
        if tots:
            got = sum(int(x.split("/")[0]) for x in tots)
            den = sum(int(x.split("/")[1]) for x in tots)
            print(f"  → 全库数值覆盖 {got}/{den} = {got/den*100:.1f}%")

    if args.json:
        json.dump({t: {"rows": [list(r[:5]) for r in d["rows"]]}
                   for t, d in results.items()},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
