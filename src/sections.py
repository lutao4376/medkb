# -*- coding: utf-8 -*-
"""
章节检测（parse.py 和 fix_sections.py 共用这一份）。

坑：不同板块的年报目录排版不一样。
  深市/沪市主板 ： "第一节 释义............................ 7"  —— 标题和省略号同一行
  科创板(688xxx)： "第一节" / "释义.............. 7"        —— 标题被拆成两行
原来的正则只在同一行里找，导致 28 份科创板年报一节都识别不出来（全为 0 节）。
这里先把"独立成行的 第X节"和它下一行粘起来，再走同一套匹配。
"""
import re

SEC_ANY = re.compile(r"第[一二三四五六七八九十]+节")
# 注意：行首只能用 [ \t]*，不能用 \s*——\s 含换行，会让 ^\s* 从上一行开始吞，
# 匹配出 ' \n \n第一节释义 ' 这种带前导换行的结果，后面 startswith 就对不上了。
# 目录页里的一条：标题 + 省略号 + 页码
SEC_TOC = re.compile(r"^[ \t]*(第[一二三四五六七八九十]+节[ \t]*[^\n.…]{2,30})[.… \t]*(\d{1,3})?[ \t]*$", re.M)
# 正文里独立成行的节标题。排掉两类噪声：
#   "第八节财务报告"之"六、…"        —— 含引号
#   "第二节公司简介和主要财务指标........12" —— 目录行，含省略号
SEC_LINE = re.compile(r"^[ \t]*(第[一二三四五六七八九十]+节[^\n”\"’’.…]{2,30})[ \t]*$", re.M)
# 整行只有 "第X节"，后面什么都没跟
JOIN = re.compile(r"^([ \t]*)(第[一二三四五六七八九十]+节)[ \t]*$\n", re.M)


def norm(t):
    """把独占一行的 "第X节" 与下一行粘连，抹平模板差异。"""
    return JOIN.sub(r"\1\2", t)


def detect_from_pages(page_texts):
    """page_texts[i] = 第 i+1 页的文本。返回 [{"title","page"}]。"""
    toc_idx, titles = None, []
    for pi, txt in enumerate(page_texts[:20]):
        t = norm(txt)
        if len(SEC_ANY.findall(t)) < 3:
            continue
        got = [m.group(1).strip() for m in SEC_TOC.finditer(t)]
        if len(got) >= 3:
            toc_idx, titles = pi, got
            break
    if toc_idx is None:
        return [], "未找到目录页"

    out, cursor, missed = [], 0, []
    for t in titles:
        key = t.replace(" ", "")
        for pi in range(cursor, len(page_texts)):
            for line in SEC_LINE.findall(norm(page_texts[pi])):
                if line.replace(" ", "").startswith(key):
                    out.append({"title": t, "page": pi + 1})
                    cursor = pi
                    break
            else:
                continue
            break
        else:
            missed.append(t)
    return out, (f"目录 {len(titles)} 条，正文定位到 {len(out)}，未定位 {missed}" if missed else "")


def detect(doc):
    """从 pymupdf Doc 里检测。"""
    texts = [doc[i].get_text() for i in range(len(doc))]
    return detect_from_pages(texts)
