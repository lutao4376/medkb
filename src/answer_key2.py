# -*- coding: utf-8 -*-
"""
判分基准重建：只认「第二节 主要会计数据」那张表。

旧版（answer_key.py）是"找锚点标签 → 往后扫 15 块"，被 p9 的
「分季度主要财务数据」截胡了：那张表同样有"归属于上市公司股东的净利润"这个标签，
扫到的第一个数 92,558,116.80 是**爱博 2025 年一季度**的数，被当成全年数写进了基准
（真值 268,170,112.64）。锚点扫法对"同名标签在附近出现多次"没有抵抗力。

改成：定位表头含 2025年/2024年 两列的表格，按行首标签取值。
取不到就退回第三节的叙述句（"本报告期公司实现营业收入 X 元，同比增长 Y%"）。
两条路都取到时互相校验，不一致就报出来。

用法:  python src/answer_key2.py            # 打印 + 对拍旧基准
       python src/answer_key2.py --write    # 写 data/answer_key.json
"""
import argparse
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CJK = r"一-鿿、。（），"

F_REV = "营业收入"
F_NP = "归属于上市公司股东的净利润"
F_DNP = "归属于上市公司股东的扣除非经常性损益的净利润"


def despace(t):
    """抹掉汉字之间的空白（PDF 把表头标签折行了），但数字之间的空白绝不动。"""
    t = re.sub(r"[ \t　]+", " ", t)
    return re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", t)


def rows_of(text):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            out.append([c.strip() for c in s.strip("|").split("|")])
    return out


def year_cols(rows):
    """表头里 2025年 / 2024年 两列的下标。"""
    for r in rows[:8]:
        i25 = next((i for i, c in enumerate(r) if re.fullmatch(r"2025\s*年", c)), None)
        i24 = next((i for i, c in enumerate(r) if re.fullmatch(r"2024\s*年", c)), None)
        if i25 is not None and i24 is not None:
            return i25, i24
    return None


FIELDS = [(F_REV, "revenue"), (F_NP, "np"), (F_DNP, "dnp")]


def num(c):
    c = c.replace(",", "").replace(" ", "")
    if not re.fullmatch(r"-?\d+(\.\d+)?%?", c):
        return None
    return float(c.rstrip("%"))


def is_money(c):
    """
    金额，不是比率。判据：没有百分号，且绝对值 >= 1000。
    财报主要会计数据一律以元为单位，所以这条能干净地把 增减(%) 和
    每股收益/净资产收益率 这些小数字挡在外面。
    """
    if "%" in c:
        return False
    v = num(c)
    return v is not None and abs(v) >= 1000


def _unit(rest):
    """字段名后面只允许挂一个单位后缀，如 （元） / (元/股) / 空。"""
    return rest == "" or re.fullmatch(r"[（(][^（()）]{0,8}[）)]", rest) is not None


def field_of(label):
    """
    ("exact", key) / ("prefix", None) / (None, None)。

    单位后缀必须整体匹配：`营业收入（元）归属于上市公司股东的净利润（元）`
    这种拼接结果同样以「营业收入（」开头，宽松前缀匹配会把它当成 complete，
    顺带把下一个字段名吞掉——迈瑞/乐普/开立/鱼跃/大博就是这么掉出去的。
    """
    if not label:
        return None, None
    for pre, key in FIELDS:
        if label.startswith(pre) and _unit(label[len(pre):]):
            return "exact", key
        if pre.startswith(label):
            return "prefix", None
    return None, None


def stitch(rows):
    """
    把表格还原成「标签-数字」交替的流再拼标签。

    为什么不能按列下标取数：三种排版都会让表头列和数据行对不上——
      · 标签折行：乐普「归属于上市公司股东的扣除非经常性」/「损益的净利润（元）」
      · 单元格错位：开立、鱼跃的表头 `| | | | 2025 年 | | | 2024 年 |`，
        表头说数字在第 4 列，数据行却写在第 3 列
      · 标签被数字夹在中间：爱博「归属于上市公司股」→ 数字 →「东的净利润」

    所以改成：把每个单元格按行序摊平，金额单独成流；标签遇到金额就先赊着，
    后面还有标签片段就继续往上拼，拼成完整字段名才认。这样上面三种排版都能过。
    """
    toks = []
    for r in rows:
        for c in r:
            c = c.strip()
            if not c:
                continue
            if is_money(c):
                toks.append(("m", c))
            elif num(c) is not None:
                # 比率（-45.50、6.22%）等：是值不是标签。当标签处理会把
                # 上一行的标签链切断，而带 % 的列恰好夹在金额和下一个标签中间。
                toks.append(("n", c))
            else:
                toks.append(("l", re.sub(r"\s+", "", despace(c))))

    out, cur, row = {}, "", None          # row = [label, [nums]]

    def commit(lab, nums):
        st, key = field_of(lab)
        if st == "exact" and key not in out and len(nums) >= 2:
            out[key] = tuple(nums[:2])

    for kind, t in toks:
        if kind == "n":
            continue                       # 比率：不打断标签链，也不进金额流
        if kind == "m":
            v = num(t)
            if row is not None:
                if len(row[1]) < 2:
                    row[1].append(v)
            elif cur:                      # 没有标签打头的数字：跳过（别的表的数）
                row = [cur, [v]]
                cur = ""
        else:
            if row is not None:
                cand = row[0] + t
                st, _ = field_of(cand)
                if st == "exact":
                    commit(cand, row[1])
                    row, cur = None, ""
                elif st == "prefix":
                    row[0] = cand
                else:
                    commit(row[0], row[1])   # 拼不上了，先看原来那个是否已经完整
                    row = None
                    cur = t if field_of(t)[0] else ""
            else:
                c2 = cur + t
                if field_of(c2)[0]:
                    cur = c2
                else:
                    cur = t if field_of(t)[0] else ""
    if row is not None:
        commit(row[0], row[1])
    return out


def from_table(chunks):
    """从 2025 年报「第二节 主要会计数据」表里取三个数。"""
    cands = []
    for c in chunks:
        if "分季度" in c["text"] or "第一季度" in c["text"]:
            continue                      # 就是这张表把旧版带沟里的
        # 不要求块里有"主要会计数据"四个字：迈瑞那类表格里只有一行表头，
        # 「五、主要会计数据和财务指标」的标题落在**上一个**正文块里，
        # 按关键词过滤会把它整块漏掉。改按章节定位。
        if "第二节" not in c["section"]:
            continue
        got = stitch(rows_of(c["text"]))
        if "revenue" in got and "np" in got:
            cands.append((c["ctype"] != "table", got, c["cite"]))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0])        # 真表格块优先于 [表格] 文本块
    return cands[0][1], cands[0][2]


def from_prose(chunks):
    """退回第三节的叙述句。"""
    pat = {
        "revenue": re.compile(r"营业收入([\d,]+\.\d\d)元[，,]?\s*同比(增长|减少)([\d.]+)%"),
        "np": re.compile(r"归属于(?:上市公司股东|母公司所有者)的净利润([\d,]+\.\d\d)元"
                         r"[，,]?\s*同比(增长|减少)([\d.]+)%"),
        "dnp": re.compile(r"扣除非经常性损益(?:后)?的净利润([\d,]+\.\d\d)元"
                          r"[，,]?\s*同比(增长|减少)([\d.]+)%"),
    }
    got = {}
    for c in chunks:
        if c["kind"] != "年度" or c["year"] != "2025" or "第三节" not in c["section"]:
            continue
        t = despace(c["text"])
        for key, p in pat.items():
            if key in got:
                continue
            m = p.search(t)
            if m:
                v = float(m.group(1).replace(",", ""))
                g = float(m.group(3)) * (1 if m.group(2) == "增长" else -1)
                got[key] = (v, g)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--chunks", default=os.path.join(ROOT, "data", "index", "chunks.jsonl"))
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
    by_co = {}
    for c in chunks:
        if c["kind"] == "年度" and c["year"] == "2025":
            by_co.setdefault(c["name"], []).append(c)

    old_p = os.path.join(ROOT, "data", "answer_key.json")
    old = json.load(open(old_p, encoding="utf-8")) if os.path.exists(old_p) else {}

    key, bad = {}, []
    for co in sorted(by_co):
        cs = by_co[co]
        tab, cite = from_table(cs)
        pro = from_prose(cs)
        # 顶层仍保留老的"亿元"扁平字段：eval_retrieval.py 用 key[n]["revenue_2025"]
        # 排序，用 key[n]["raw"] 里的元级值做匹配。字段名换了会直接 KeyError
        # 或者静默退化成"取不到值"（覆盖度全 0），所以两种都写。
        rec = {"unit": "元", "raw": {}, "growth": {}, "src": cite or "-"}
        for f in ("revenue", "np", "dnp"):
            if tab and f in tab:
                v, v24 = tab[f]
                rec["raw"][f + "_2025"] = v
                rec["raw"][f + "_2024"] = v24
            elif f in pro:
                rec["raw"][f + "_2025"] = pro[f][0]
            if f in pro:
                rec["growth"][f] = pro[f][1]
            elif tab and f in tab:
                a, b = tab[f]
                rec["growth"][f] = round((a / b - 1) * 100, 2) if b else None
        for f in ("revenue", "np", "dnp"):
            for yr in ("2025", "2024"):
                v = rec["raw"].get(f"{f}_{yr}")
                if v is not None:
                    rec[f"{f}_{yr}"] = round(v / 1e8, 2)      # 亿元，给人看的
        # 表格 vs 叙述句 对拍
        for f in ("revenue", "np", "dnp"):
            if tab and f in tab and f in pro:
                if abs(tab[f][0] - pro[f][0]) > 1:
                    bad.append(f"{co}.{f}: 表={tab[f][0]:,} 叙述={pro[f][0]:,}")
        key[co] = rec

    print(f"{'公司':<10} {'营业收入':>18} {'归母净利润':>18} {'扣非':>18}  {'营收同比':>9} {'归母同比':>9}  来源")
    for co, r in key.items():
        g = r["growth"]
        print(f"{co:<10} {r['raw'].get('revenue_2025', 0):18,.2f} "
              f"{r['raw'].get('np_2025', 0):18,.2f} "
              f"{r['raw'].get('dnp_2025') or 0:18,.2f}  "
              f"{g.get('revenue') or 0:8.2f}% {g.get('np') or 0:8.2f}%  {r['src'][:34]}")

    print("\n—— 与旧基准的差异（旧基准是错的那些）——")
    for co, r in key.items():
        o = old.get(co, {}).get("raw", {})
        for f in ("revenue", "np", "dnp"):
            a, b = r["raw"].get(f + "_2025"), o.get(f + "_2025")
            if a and b and abs(a - b) > 1:
                print(f"  {co}.{f}: 旧 {b:,.2f}  ->  新 {a:,.2f}  ({(a/b-1)*100:+.1f}%)")
            elif a and not b:
                print(f"  {co}.{f}: 旧 缺失      ->  新 {a:,.2f}")

    if bad:
        print("\n⚠ 表格与叙述句不一致：")
        for x in bad:
            print("   ", x)
    else:
        print("\n表格与叙述句全部对得上。")

    if args.write:
        json.dump(key, open(old_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n-> {old_p}")


if __name__ == "__main__":
    main()
