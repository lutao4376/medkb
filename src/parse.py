# -*- coding: utf-8 -*-
"""
48 份财报 PDF -> 结构化 JSON。

每份输出 data/parsed/{code}_{name}_{year}{kind}.json：
{
  "meta":     {code,name,sector,year,kind,pdf,pages,chars,tables},
  "sections": [{title, page}],          # 第X节，用于切块时打"章节"标签
  "pages":    [{page, section, text}],  # text 里表格已还原成 markdown 表格
}
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdftable import page_tables, looks_tabular, rows_to_markdown, page_to_rows  # noqa: E402
from sections import detect as detect_sections  # noqa: E402


def parse_pdf(path, meta):
    doc = pymupdf.open(path)
    sections, _note = detect_sections(doc)

    pages = []
    n_tables = 0
    n_chars = 0
    for pi in range(len(doc)):
        page = doc[pi]
        raw = page.get_text()
        tables = page_tables(page)
        n_tables += len(tables)

        # 三大表这类无框线大表：整页就是这张表，用还原结果替换线性文字，避免重复
        big = [(k, r) for k, r in tables if k == "coords" and len(r) >= 15]
        if big:
            body = "\n\n".join(f"[表格]\n{rows_to_markdown(r)}" for _, r in big)
        else:
            body = raw.strip()
            for k, r in tables:
                body += f"\n\n[表格]\n{rows_to_markdown(r)}"

        if not body.strip():
            continue
        n_chars += len(body)

        # 这一页属于哪一节
        sec = ""
        for s in sections:
            if s["page"] <= pi + 1:
                sec = s["title"]
            else:
                break
        pages.append({"page": pi + 1, "section": sec, "text": body})

    doc.close()
    return {
        "meta": {**meta, "pdf": os.path.basename(path), "pages": len(pages),
                 "chars": n_chars, "tables": n_tables},
        "sections": sections,
        "pages": pages,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    meta_all = json.load(open(META_PATH, encoding="utf-8"))

    jobs = []
    for code, m in meta_all.items():
        if args.only and code != args.only:
            continue
        for key, rec in m["reports"].items():
            pdf = os.path.join(PDF_DIR, os.path.basename(rec["path"]))
            if os.path.exists(pdf):
                jobs.append((code, m, key, rec, pdf))
    jobs.sort(key=lambda j: (j[0], j[2]))
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"待解析 {len(jobs)} 份")
    t0 = time.time()
    total_ok = 0
    for i, (code, m, key, rec, pdf) in enumerate(jobs, 1):
        out = os.path.join(OUT_DIR, f"{code}_{m['name']}_{rec['year']}{rec['kind']}报告.json")
        try:
            data = parse_pdf(pdf, {"code": code, "name": m["name"], "sector": m["sector"],
                                   "year": rec["year"], "kind": rec["kind"]})
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            mt = data["meta"]
            print(f"  [{i}/{len(jobs)}] {mt['name']} {mt['year']}{mt['kind']}  "
                  f"{mt['pages']}页 {mt['chars']//10000}万字 {mt['tables']}表  "
                  f"{len(data['sections'])}节  {os.path.getsize(out)/1e6:.1f}MB")
            total_ok += 1
        except Exception as e:
            print(f"  [{i}/{len(jobs)}] !! {os.path.basename(pdf)}: {type(e).__name__} {e}")
    print(f"\n完成 {total_ok}/{len(jobs)} 份，用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
