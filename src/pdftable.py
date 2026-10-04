# -*- coding: utf-8 -*-
"""
财报 PDF 表格还原。

年报里的表分两种，用一套策略治不了：
  * 有框线的表（董监高情况、股东情况…）—— find_tables('lines') 质量最好
  * 无框线的表（**合并资产负债表/利润表/现金流量表**）—— find_tables 全部漏掉，
    'text' 策略能找到但会把"营业收入"切成"营业"/"收入"（PDF 里两字间距偏大）

这里自己按坐标重建：词 -> 按 y 聚成行 -> 按 x 空档切成列。
"""
import re

import pymupdf

NUM_RE = re.compile(r"^[（(]?[-−—]?[\d,，.]+[）)]?%?$|^[-−—]$")


def _rows_of(words, y_tol=4.0):
    """按 y 中心把词聚成行，返回 [(y_center, [word,...]), ...]。"""
    ws = sorted(words, key=lambda w: ((w[1] + w[3]) / 2.0, w[0]))
    rows = []
    for w in ws:
        yc = (w[1] + w[3]) / 2.0
        if rows and abs(yc - rows[-1][0]) <= y_tol:
            rows[-1][1].append(w)
        else:
            rows.append((yc, [w]))
    return rows


def _cluster(vals, tol):
    """一维聚类，返回簇中心（升序）。"""
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(c) / len(c) for c in out]


def _column_rights(words, tol=7.0):
    """数字列的锚点 = 数字词的右边缘（财务数字都是右对齐的）。"""
    rights = [w[2] for w in words if NUM_RE.match(w[4].replace(" ", ""))]
    if len(rights) < 8:
        return []
    return _cluster(rights, tol)


def _assign(w, rights):
    """
    归属：数字列按右边缘锚点就近；标签列是第一个数字列左边的那条带。
    返回列下标，0 = 标签列。
    """
    r1 = rights[0]
    band0 = r1 - 40.0                      # 标签列右界（附注号很短，留 40pt 足够）
    cx = (w[0] + w[2]) / 2.0
    if cx <= band0:
        return 0
    if len(rights) == 1:
        return 1
    bounds = [(rights[i] + rights[i + 1]) / 2.0 for i in range(len(rights) - 1)]
    for i, b in enumerate(bounds):
        if cx <= b:
            return i + 1
    return len(rights)


def page_to_rows(page, y_tol=4.0, min_cols=3, min_rows=4):
    """把一页还原成 list[list[str]]；不像表就返回 None。"""
    words = [w for w in page.get_text("words") if w[4].strip()]
    if len(words) < 15:
        return None
    rights = _column_rights(words)
    if len(rights) < 2:
        return None

    out = []
    for yc, ws in _rows_of(words, y_tol=y_tol):
        ws = sorted(ws, key=lambda w: w[0])
        cells = [""] * (len(rights) + 1)
        for w in ws:
            ci = _assign(w, rights)
            cells[ci] = (cells[ci] + " " + w[4]).strip() if cells[ci] else w[4]
        out.append(cells)

    out = [r for r in out if any(c.strip() for c in r)]
    if len(out) < min_rows:
        return None
    # 去掉整列全空的（锚点聚类可能多出空列）
    keep = [i for i in range(len(rights) + 1) if sum(1 for r in out if r[i].strip()) >= 2]
    if len(keep) < min_cols:
        return None
    return [[r[i] for i in keep] for r in out]


def looks_tabular(rows):
    """这一页还原出来像不像表：数字占比够高才算。"""
    if not rows or len(rows) < 4:
        return False
    cells = [c for r in rows for c in r]
    nums = sum(1 for c in cells if NUM_RE.match(c.replace(" ", "")))
    return nums >= 6 and nums / max(1, len(cells)) >= 0.12


def page_tables(page):
    """
    返回这一页的表，优先用带框线的（质量高），没有再退回坐标重建。
    返回 [(kind, rows), ...]，kind ∈ {'lines', 'coords'}
    """
    res = []
    try:
        for tb in page.find_tables(strategy="lines").tables:
            rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in tb.extract()]
            rows = [r for r in rows if any(r)]
            if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                res.append(("lines", rows))
    except Exception:
        pass
    if res:
        # 有框线表时，仍然看看这页有没有坐标版的大表（三大表可能和框线表同页）
        pass

    try:
        rows = page_to_rows(page)
        if rows and looks_tabular(rows):
            res.append(("coords", rows))
    except Exception:
        pass
    return res


def rows_to_markdown(rows):
    """表 -> markdown，让它在文本块里也是"一行一个科目"。"""
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    out = []
    for i, r in enumerate(rows):
        r = list(r) + [""] * (w - len(r))
        out.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
        if i == 0:
            out.append("|" + "---|" * w)
    return "\n".join(out)
