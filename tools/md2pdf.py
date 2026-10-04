# -*- coding: utf-8 -*-
"""
Markdown -> PDF，用来把「一页结论」排成能直接交的版。

没有 pandoc，所以走 python-markdown 渲染成 HTML，再让无头 Chrome 的
--print-to-pdf 打印。这条路的好处是中文断行、表格、`<details>` 都按浏览器来，
和我截图看到的是同一个渲染引擎，不会出现"md 里好好的、导出就散架"。

用法:
    python tools/md2pdf.py docs/一页结论.md            # -> docs/一页结论.pdf
    python tools/md2pdf.py docs/一页结论.md --out docs/结论.pdf --size A4
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: %(size)s; margin: 11mm 11mm 10mm 11mm; }
* { box-sizing: border-box; }
body { font: 9.5pt/1.36 "Microsoft YaHei","Segoe UI",sans-serif; color:#1a1a1a;
       margin:0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
h1 { font-size: 14pt; margin: 0 0 1mm; letter-spacing:-0.2px; }
h2 { font-size: 11pt; margin: 2.1mm 0 0.8mm; padding-bottom:0.5mm;
     border-bottom: 1.5px solid #d8d8d8; }
h3 { font-size: 10pt; margin: 2mm 0 0.6mm; color:#333; }
p, li { margin: 0.6mm 0; }
ul, ol { padding-left: 5mm; margin: 0.6mm 0; }
table { border-collapse: collapse; width: 100%%; margin: 1mm 0; font-size: 8.5pt; }
th, td { border: 0.5pt solid #c9c9c9; padding: 0.6mm 1.4mm; text-align: left;
         vertical-align: top; }
th { background: #f2f4f7; font-weight: 600; }
code { font-family: Consolas,"Courier New",monospace; font-size: 9pt;
       background: #f4f4f6; padding: 0 1px; border-radius: 2px; }
pre { background: #f7f7f9; border-left: 2.5pt solid #c9c9c9; padding: 2mm 3mm;
      font-size: 8.5pt; line-height:1.5; overflow-wrap: anywhere; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
blockquote { margin: 2mm 0; padding: 1mm 3mm; border-left: 2.5pt solid #b9c6d4;
             background: #f7f9fb; color:#333; }
hr { border: none; border-top: 0.5pt solid #ddd; margin: 4mm 0; }
a { color: #1a56a8; text-decoration: none; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
tr, pre, blockquote { break-inside: avoid; page-break-inside: avoid; }
""" % {"size": "A4"}

HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>%(title)s</title><style>%(css)s</style></head>
<body>%(body)s</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", default="A4", choices=["A4", "A3", "Letter"])
    ap.add_argument("--keep-html", action="store_true", help="留一份 .html 方便对版")
    args = ap.parse_args()

    import markdown

    src = args.md if os.path.isabs(args.md) else os.path.join(ROOT, args.md)
    if not os.path.exists(src):
        raise SystemExit(f"没有这个文件：{src}")
    text = open(src, encoding="utf-8").read()

    # md 里手写的 <details> 要原样透传，python-markdown 默认会把块内的
    # markdown 当 raw HTML 丢掉；剩下的正文走 extensions。
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
        output_format="html5")

    title = os.path.splitext(os.path.basename(src))[0]
    html = HTML % {"title": title,
                   "css": CSS.replace("size: A4;", f"size: {args.size};"),
                   "body": body}

    out_pdf = args.out or os.path.splitext(src)[0] + ".pdf"
    if not os.path.isabs(out_pdf):
        out_pdf = os.path.join(ROOT, out_pdf)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    exe = next((p for p in BROWSERS if os.path.exists(p)), None)
    if not exe:
        raise SystemExit("找不到 Chrome/Edge，没法打印 PDF")

    tmp = tempfile.mkdtemp(prefix="md2pdf-")
    try:
        h = os.path.join(tmp, "doc.html")
        open(h, "w", encoding="utf-8").write(html)
        if args.keep_html:
            shutil.copyfile(h, os.path.splitext(out_pdf)[0] + ".html")

        # 打印到 %TEMP% 再搬过去：Chrome 对带中文/空格的目标路径偶发写不进去
        p = os.path.join(tmp, "out.pdf")
        r = subprocess.run(
            [exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={p}", "--virtual-time-budget=8000",
             f"--user-data-dir={os.path.join(tmp, 'prof')}",
             "file:///" + h.replace("\\", "/")],
            capture_output=True, text=True, timeout=180)
        for _ in range(40):                       # Chrome 有时不等写完就返回
            if os.path.exists(p) and os.path.getsize(p) > 0:
                break
            time.sleep(0.25)
        if not os.path.exists(p):
            sys.stderr.write((r.stderr or "") + "\n")
            raise SystemExit("Chrome 没有产出 PDF")
        shutil.move(p, out_pdf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 页数用 PyMuPDF 数（本仓库解析 PDF 本来就装了）；数不出来就只报大小。
    # 别用正则扫 PDF 字节：Chrome 输出的是压缩对象流，扫不到 /Type /Page。
    try:
        import pymupdf as fitz
        n = fitz.open(out_pdf).page_count
    except Exception:
        n = None
    size = f"{os.path.getsize(out_pdf)/1024:.0f} KB"
    pages = f"{n} 页" if n else "页数未数"
    flag = "  ⚠ 超出一页" if n and n > 1 else ""
    print(f"-> {out_pdf}  ({size}, {pages}){flag}")


if __name__ == "__main__":
    main()
