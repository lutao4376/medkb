# -*- coding: utf-8 -*-
"""
把题目集里带数字的 expected 从 answer_key.json 现算出来，不再手打。

起因：手打的真值悄悄飘了——Q8 写"澳华 -47.62%、爱博 -23.14%"，实际是
-45.50% 和 -30.96%（爱博那条还是错基准的连带产物）。数字一旦有两处来源
就一定会有对不上的那天，所以只留一处：key 里。

用法:  python src/make_questions.py             # 只打印
       python src/make_questions.py --write
"""
import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    K = json.load(open(os.path.join(ROOT, "data", "answer_key.json"), encoding="utf-8"))
    qp = os.path.join(ROOT, "data", "questions.json")
    qs = json.load(open(qp, encoding="utf-8"))

    rev = lambda n: K[n]["raw"]["revenue_2025"]
    np_ = lambda n: K[n]["raw"]["np_2025"]
    dnp = lambda n: K[n]["raw"]["dnp_2025"]
    gr = lambda n, f: K[n]["growth"][f]
    yi = lambda v: v / 1e8

    order = sorted(K, key=lambda n: -rev(n))
    q4 = sorted(K, key=lambda n: -np_(n) / rev(n))
    q5 = sorted([n for n in K if gr(n, "revenue") < 0], key=lambda n: gr(n, "revenue"))
    q8 = sorted([n for n in K if gr(n, "np") < 0], key=lambda n: gr(n, "np"))

    E = {
        "Q1": f"营业收入 {yi(rev('迈瑞医疗')):.2f} 亿元；归母净利润 {yi(np_('迈瑞医疗')):.2f} 亿元；"
              f"扣非归母净利润 {yi(dnp('迈瑞医疗')):.2f} 亿元（均为 2025 年度合并口径）",
        "Q2": f"营业收入同比增长 {gr('联影医疗','revenue'):.2f}%；"
              f"归母净利润同比增长 {gr('联影医疗','np'):.2f}%",
        "Q3": "、".join(f"{n} {yi(rev(n)):.2f} 亿" for n in order[:6])
              + "（按 2025 年度合并口径排序）",
        "Q4": f"{q4[0]}最高，{np_(q4[0])/rev(q4[0])*100:.2f}%"
              f"（归母净利润 {yi(np_(q4[0])):.2f} 亿 ÷ 营业收入 {yi(rev(q4[0])):.2f} 亿）；"
              + "、".join(f"{n} {np_(n)/rev(n)*100:.2f}%" for n in q4[1:4]),
        "Q5": f"只有 {len(q5)} 家下滑：" + "、".join(
            f"{n}（{yi(rev(n)):.2f} 亿，同比 {gr(n,'revenue'):.2f}%）" for n in q5)
              + f"；{q5[0]}降幅最大",
        "Q8": f"共 {len(q8)} 家：" + "、".join(
            f"{n}（{yi(np_(n)):.2f} 亿，{gr(n,'np'):.2f}%）" for n in q8),
        "Q9": f"须区分口径：归母净利润 {yi(np_('大博医疗')):.2f} 亿元"
              f"（{np_('大博医疗'):,.2f} 元）；扣非归母净利润 {yi(dnp('大博医疗')):.2f} 亿元"
              f"（{dnp('大博医疗'):,.2f} 元）",
    }
    # Q7 的数字在「主营业务分产品/分地区」表里，也在第三节正文里；两个来源都核过
    E["Q7"] = ("齿科类产品 7,113.01 万元（同比 +8.04%）；境外业务 31,198.44 万元（同比 +43.81%）"
               "——「主营业务分产品/分地区」表与第三节正文两处都披露，数值一致")

    fixed_notes = {
        "Q7": "数字在「主营业务分产品/分地区」表里，第三节正文也复述了一遍，用来看正文块与表格块哪一边先被召回。",
    }

    changed = []
    for q in qs:
        if q["id"] in E and q.get("expected") != E[q["id"]]:
            changed.append((q["id"], q.get("expected"), E[q["id"]]))
            q["expected"] = E[q["id"]]
        if q["id"] in fixed_notes:
            q["note"] = fixed_notes[q["id"]]

    for qid, old, new in changed:
        print(f"■ {qid}\n  旧: {old}\n  新: {new}\n")

    if args.write:
        json.dump(qs, open(qp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"-> {qp}（改了 {len(changed)} 题的 expected）")


if __name__ == "__main__":
    main()
