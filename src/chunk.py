# -*- coding: utf-8 -*-
"""
解析结果 -> 文本块。

对齐课上做法：800 字一块、重合 120 字、尽量在句末收尾，每块带
[公司, 章节, 页码, 块序号] 四个标签。

一个必须避开的坑（课上伊利那题就栽在这）：**表格被切开后，表名和表头留在了
前一块里，这一块只剩数字，模型分不出是哪个口径**。所以这里
  ① 表格尽量整块保留；
  ② 非切不可时，每个碎片都重新带上表头，并且前置一句带公司/页面的表格说明。
"""
import argparse
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED_DIR = os.path.join(ROOT, "data", "parsed")
OUT_PATH = os.path.join(ROOT, "data", "chunks.jsonl")

SIZE = 800       # 每块目标字数
OVERLAP = 120    # 相邻块重合
MIN_TAIL = 200   # 尾块太短就并进上一块

SENT_END = "。！？；：\n"


def split_units(text):
    """一页拆成若干单元：('text', 段落) 或 ('table', 表格markdown)。表格总在页面末尾。"""
    parts = re.split(r"\n\n\[表格\]\n", text)
    units = []
    body = parts[0].strip()
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if para:
            units.append(("text", para))
    for tb in parts[1:]:
        tb = tb.strip()
        if tb:
            units.append(("table", tb))
    return units


def _tail_split(s, size):
    """尽量在句末断开。"""
    if len(s) <= size:
        return s, ""
    cut = s.rfind("\n", 0, size)
    if cut < size * 0.5:
        for ch in SENT_END:
            i = s.rfind(ch, 0, size)
            if i > size * 0.5:
                cut = i + 1
                break
        else:
            cut = size
    return s[:cut], s[cut:]


def pack_text(units, size=SIZE, overlap=OVERLAP):
    """把纯文字单元打包成块（带重合）。"""
    buf = ""
    out = []
    for _, para in units:
        while len(para) > size:
            head, para = _tail_split(para, size)
            buf = (buf + "\n" + head).strip() if buf else head
            out.append(buf)
            buf = out[-1][-overlap:] if overlap else ""
        if not buf:
            buf = para
        elif len(buf) + len(para) + 1 <= size:
            buf += "\n" + para
        else:
            out.append(buf)
            buf = (buf[-overlap:] + "\n" + para).strip()
    if buf.strip():
        out.append(buf)
    # 尾块太短就并进上一块
    if len(out) >= 2 and len(out[-1]) < MIN_TAIL:
        out[-2] = out[-2] + "\n" + out[-1]
        out.pop()
    return out


def pack_table(md, size=SIZE):
    """
    表格：整块保留优先；太大才按行切，且每片都带表头 + 说明。
    返回 [块文本]（不含说明前缀，调用方补）。
    """
    if len(md) <= size * 1.6:
        return [md]
    lines = [ln for ln in md.split("\n") if ln.strip()]
    if len(lines) < 3:
        return [md]
    head = lines[:2]                      # 表头 + 分隔行
    rows = lines[2:]
    out, cur = [], list(head)
    for r in rows:
        if len("\n".join(cur + [r])) > size and len(cur) > len(head):
            out.append("\n".join(cur))
            cur = list(head)              # 每片都重新带上表头
        cur.append(r)
    if len(cur) > len(head) or len(out) == 0:
        out.append("\n".join(cur))
    return out


def chunk_file(path, meta):
    data = json.load(open(path, encoding="utf-8"))
    m = data["meta"]
    head = f"{m['name']}（{m['code']}）{m['year']}年{m['kind']}报告"
    chunks = []
    for pg in data["pages"]:
        sec = pg["section"] or ""
        page = pg["page"]
        units = split_units(pg["text"])
        texts = [u for u in units if u[0] == "text"]
        tables = [u for u in units if u[0] == "table"]

        pieces = [(t, "text") for t in pack_text(texts)]
        for _, md in tables:
            for t in pack_table(md):
                pieces.append((t, "table"))

        for i, (t, ctype) in enumerate(pieces, 1):
            if len(t.strip()) < 30:
                continue
            chunks.append({
                "code": m["code"], "name": m["name"], "sector": m["sector"],
                "year": m["year"], "kind": m["kind"], "section": sec,
                "page": page, "seq": i, "ctype": ctype,
                "text": t,
                # 向量化时把公司名+章节名一起放进去（课上做法），检索才分得清是谁的数
                "embed_text": f"{head} {sec} 第{page}页\n{t}",
                "cite": f"{m['name']} {m['year']}年{m['kind']}报告 第{page}页",
            })
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(PARSED_DIR) if f.endswith(".json"))
    if args.limit:
        files = files[: args.limit]
    print(f"待切块 {len(files)} 份")

    all_chunks = []
    for i, f in enumerate(files, 1):
        try:
            cs = chunk_file(os.path.join(PARSED_DIR, f), None)
            all_chunks.extend(cs)
            n_tab = sum(1 for c in cs if c["ctype"] == "table")
            if i % 8 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] 累计 {len(all_chunks)} 块")
        except Exception as e:
            print(f"  !! {f}: {type(e).__name__} {e}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_tab = sum(1 for c in all_chunks if c["ctype"] == "table")
    lens = sorted(len(c["text"]) for c in all_chunks)
    print(f"\n共 {len(all_chunks)} 块 -> {OUT_PATH}")
    print(f"  表格块 {n_tab}（{n_tab/max(1,len(all_chunks))*100:.0f}%）  文字块 {len(all_chunks)-n_tab}")
    print(f"  块长 中位 {lens[len(lens)//2]}  最短 {lens[0]}  最长 {lens[-1]}")


if __name__ == "__main__":
    main()
