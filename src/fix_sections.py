# -*- coding: utf-8 -*-
"""
只重跑章节检测，不重做表格还原（那步一次要 9 分钟）。
复用 sections.py 的新逻辑，改写已解析 JSON 里的 sections 和每页的 section 字段。
"""
import glob
import json
import os
import sys
import time

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sections import detect_from_pages  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSED = os.path.join(ROOT, "data", "parsed")
PDF = os.path.join(ROOT, "data", "pdf")


def main():
    files = sorted(glob.glob(os.path.join(PARSED, "*.json")))
    print(f"共 {len(files)} 份")
    t0 = time.time()
    fixed, still_zero = 0, []
    for i, f in enumerate(files, 1):
        d = json.load(open(f, encoding="utf-8"))
        m = d["meta"]
        pdf = os.path.join(PDF, m["pdf"])
        if not os.path.exists(pdf):
            print(f"  !! 缺 PDF: {m['pdf']}")
            continue
        doc = pymupdf.open(pdf)
        texts = [doc[k].get_text() for k in range(len(doc))]
        doc.close()

        secs, note = detect_from_pages(texts)
        d["sections"] = secs
        for pg in d["pages"]:
            s = ""
            for x in secs:
                if x["page"] <= pg["page"]:
                    s = x["title"]
                else:
                    break
            pg["section"] = s
        json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False)

        if secs:
            fixed += 1
            print(f"  [{i}/{len(files)}] {m['name']} {m['year']}{m['kind']}  "
                  f"{len(secs)}节  首节第{secs[0]['page']}页  {note}")
        else:
            still_zero.append(f"{m['name']} {m['year']}{m['kind']}")
            print(f"  [{i}/{len(files)}] {m['name']} {m['year']}{m['kind']}  !! 仍为 0 节  {note}")

    print(f"\n有章节 {fixed}/{len(files)}，仍为 0 节 {len(still_zero)} 份，"
          f"用时 {(time.time()-t0)/60:.1f} 分")
    for s in still_zero:
        print("   ", s)


if __name__ == "__main__":
    main()
