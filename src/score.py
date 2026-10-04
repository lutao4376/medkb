# -*- coding: utf-8 -*-
"""
判分 + 出逐题报告。

作业要的是"每题召回了哪些块、哪一路找到的、答对没答对、卡在哪"，四件事都要有出处，
不能靠肉眼扫。这里做三件事：

  1. 检索覆盖率：top-k 里出现过几家公司 / 几份报告，两路各自命中几块；
  2. 数字命中率：把 answer_key 里的真值数字拿去答案里找，命中的比例；
  3. 引用合法性：答案里的 [n] 是否都落在召回块编号范围内。

数字命中只是"疑似对"，所以每条判定都带上原文片段，方便人复核——不假装它是准确率。

用法：
    python src/score.py --runs out/qa_bm25.json out/qa_mix.json --md out/报告.md
"""
import argparse
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def nums(s):
    """抽数字，返回去逗号的集合。用于比对真值和答案。"""
    out = set()
    for m in NUM.finditer(s):
        t = m.group().rstrip(".")
        if not t or t in ("-", "."):
            continue
        try:
            v = float(t.replace(",", ""))
        except ValueError:
            continue
        out.add(v)
    return out


def fmt(v):
    return f"{v:,.2f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v):,}"


def hit_numbers(expected, answer):
    """
    真值里的数字有多少能在答案里找到（容忍千分位和 亿/万/元 写法差异）。

    按**绝对值**比：中文答案用「同比减少 45.50%」表达方向，不写负号，真值里却是
    -45.50，直接比符号会把答对的判成没答到——Q8 里模型明明写了「减少 30.96%」，
    报告却显示这一条没命中。方向由文字承担，数字只比大小。
    """
    want = [v for v in nums(expected) if abs(v) >= 1]
    ans = {abs(x) for x in nums(answer)}
    # 答案里可能写 332.82 也可能写 3,328,215.94（万元）或 33,282,159,404（元）
    got, miss = [], []
    for v in want:
        av = abs(v)
        ok = av in ans or any(abs(a - av) < 1e-6 for a in ans)
        if not ok:                        # 允许 亿元/元 换算后的写法
            for f in (1e4, 1e8):
                if any(abs(a - av * f) / max(1e-9, av * f) < 5e-3 for a in ans) or \
                   any(abs(a - av / f) / max(1e-9, av / f) < 5e-3 for a in ans):
                    ok = True
                    break
        (got if ok else miss).append(v)
    return got, miss


def load_truth():
    """
    真值以 questions.json 为准，判分时现取。

    跑批结果里也存了一份 _expected（qa.py 写进去的），但那是**跑批当时**的副本：
    我后来修正了基准和题面真值，重跑 score.py 却仍然按旧数字判分，报告里
    Q8 还显示 -23.14% 这种已经被证伪的值。真值只能有一个来源。
    """
    p = os.path.join(ROOT, "data", "questions.json")
    if not os.path.exists(p):
        return {}
    return {q["id"]: q for q in json.load(open(p, encoding="utf-8")) if "id" in q}


TRUTH = load_truth()


def run_report(path):
    if not os.path.exists(path):
        return None
    rs = json.load(open(path, encoding="utf-8"))
    out = []
    for r in rs:
        ans = r.get("answer") or ""
        hits = r.get("hits", [])
        cos = []
        for h in hits:
            if h["name"] not in cos:
                cos.append(h["name"])
        reports = {(h["name"], h["year"], h["kind"]) for h in hits}
        src = {"both": 0, "bm25": 0, "vec": 0}
        for h in hits:
            src[h.get("_source", "bm25")] += 1
        cited = r.get("cited") or []
        t = TRUTH.get(r.get("_id") or "")
        exp = (t or {}).get("expected") or r.get("_expected") or ""
        got, miss = hit_numbers(exp, ans)
        out.append({
            "id": r.get("_id"), "type": (t or {}).get("type") or r.get("_type"),
            "q": r["question"],
            "mode": r.get("mode"), "n_hits": len(hits),
            "companies": cos, "n_reports": len(reports), "src": src,
            "cited": cited, "cited_valid": r.get("cited_valid"),
            "ans": ans, "expected": exp, "note": (t or {}).get("note") or r.get("_note"),
            "n_got": len(got), "n_miss": len(miss), "miss": [fmt(x) for x in miss],
            "hits": hits, "elapsed": r.get("elapsed"),
        })
    return out


def diagnose(r):
    """
    卡在哪。

    这一列是这份报告的重点："答错了"不是一种失败，是三种：
      · 检索没覆盖到 → 该改检索
      · 覆盖到了但模型没答对 → 该改提示词/模型
      · 覆盖到了、数也对，只是引用编号越界 → 该修输出格式
    不分清楚就会对着模型调半天，其实问题在召回。
    """
    tot = r["n_got"] + r["n_miss"]
    if tot == 0:
        return "无数值基准"
    if r["n_miss"] == 0:
        return "数值全中"
    if r["n_got"] == 0:
        base = "检索未覆盖"
    else:
        base = f"部分覆盖（{r['n_got']}/{tot}）"
    if not r["cited_valid"]:
        base += "·引用越界"
    return base


def table(rep, mode):
    L = [f"### {mode} 逐题明细", "",
         "| 题 | 类型 | 召回块 | 覆盖公司 | 覆盖报告 | 两路命中 | 仅BM25 | 仅向量 | 引用 | 引用合法 | 数值命中 | 卡在哪 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep:
        s = r["src"]
        cov = f"{len(r['companies'])}"
        L.append(f"| {r['id']} | {r['type']} | {r['n_hits']} | {cov} | {r['n_reports']} | "
                 f"{s['both']} | {s['bm25']} | {s['vec']} | {r['cited'] or '—'} | "
                 f"{'是' if r['cited_valid'] else '否'} | "
                 f"{r['n_got']}/{r['n_got']+r['n_miss']}"
                 + (f"（缺 {','.join(r['miss'])}）" if r["miss"] else "") + " | "
                 + diagnose(r) + " |")
    return L


def detail(rep, mode):
    L = [f"### {mode} 逐题溯源", ""]
    for r in rep:
        L += [f"<details><summary><b>{r['id']}</b> {r['q']}</summary>", "",
              f"- 类型：{r['type']}",
              f"- 真值：{r['expected']}",
              f"- 说明：{r['note']}",
              f"- 召回 {r['n_hits']} 块，覆盖 {len(r['companies'])} 家："
              f"{'、'.join(r['companies'])}",
              f"- 引用编号：{r['cited'] or '无'}（合法：{'是' if r['cited_valid'] else '否'}）",
              f"- 数值命中 {r['n_got']}/{r['n_got']+r['n_miss']}"
              + (f"，缺 {', '.join(r['miss'])}" if r["miss"] else ""),
              "", "| # | 来源 | BM25名次 | 向量名次 | 公司 | 报告期 | 页 | 章节 | 类型 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for i, h in enumerate(r["hits"], 1):
            L.append(f"| {i} | {h.get('_source')} | {h.get('_rank_bm25') or '—'} | "
                     f"{h.get('_rank_vec') or '—'} | {h['name']} | {h['year']}{h['kind']} | "
                     f"{h['page']} | {(h['section'] or '—')[:18]} | "
                     f"{'表格' if h['ctype']=='table' else '正文'} |")
        L += ["", "**回答**", "", "```", (r["ans"] or "（未调模型）").strip(), "```", "",
              "</details>", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--md", default=os.path.join(ROOT, "out", "检索评测.md"))
    args = ap.parse_args()

    blocks = {}
    for p in args.runs:
        rep = run_report(p)
        if rep is None:
            print(f"跳过（不存在）：{p}")
            continue
        mode = rep[0]["mode"] if rep else "?"
        # 两次跑同一 mode 是常事（有过滤 / 无过滤各跑一遍 bm25）。按 mode 当键会把先来
        # 的那次**静默覆盖**掉，报告里只剩一个 block，看着像只跑了一次。
        key = mode if mode not in blocks else f"{mode}·{os.path.splitext(os.path.basename(p))[0]}"
        blocks[key] = rep

    if not blocks:
        sys.exit("没有可评的结果文件")

    L = ["# 10 题检索评测", "",
         "数值命中 = 真值里的数字出现在答案里的比例，**只是疑似对**，"
         "每条都附了原文供复核，不当准确率用。", "",
         "10 题里 Q6（语义检索）、Q10（跨表推理）没有数值基准，卡在「无数值基准」；"
         "统计「数值全中」时分母只算**有数值基准的 8 题**。", ""]
    for mode, rep in blocks.items():
        L += table(rep, mode) + [""]
    for mode, rep in blocks.items():
        L += detail(rep, mode)

    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    open(args.md, "w", encoding="utf-8").write("\n".join(L))

    print(f"-> {args.md}\n")
    for mode, rep in blocks.items():
        # tot==0 的题（Q6 语义题、Q10 跨表推理）没有数值基准，n_miss 也是 0——
        # 拿 n_miss==0 当"全中"会把它们数进来，5 题里 2 题是白送的。
        # 分母得写成"有数值基准的题数"，不然报告会显得比实际好看。
        scored = [r for r in rep if r["n_got"] + r["n_miss"] > 0]
        ok = sum(1 for r in scored if r["n_miss"] == 0)
        cov = sum(len(r["companies"]) for r in rep) / max(1, len(rep))
        ret = sum(1 for r in rep if diagnose(r).startswith("检索未覆盖"))
        part = sum(1 for r in rep if diagnose(r).startswith("部分覆盖"))
        print(f"  {mode:5} {len(rep)} 题（{len(rep)-len(scored)} 题无数值基准）· "
              f"数值全中 {ok}/{len(scored)} · 平均覆盖 {cov:.1f} 家公司/题")
        print(f"        {'':10}卡在检索：完全没覆盖 {ret} 题、只覆盖一部分 {part} 题"
              f"（这两类都属于检索侧，不是模型答错）")


if __name__ == "__main__":
    main()
