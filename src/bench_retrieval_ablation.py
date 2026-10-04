# -*- coding: utf-8 -*-
"""
检索消融：把"每一层过滤/机制各值多少分"一次性量出来，不调模型。

为什么要有这个脚本：README 里那张表（5.6% → 14.6% → …）如果靠手抄，
改一次参数就得重新抄一遍，早晚抄错。这里让它一条命令现算。

     python src/bench_retrieval_ablation.py                 # 只跑不需要向量的那几档
     python src/bench_retrieval_ablation.py --with-vec      # 向量建好后连 mix/vec 一起跑

口径说明（很重要，别把它当准确率看）：
  数值覆盖 = 该题要求的那几个原始披露值，有多少个能在**召回块拼起来的文本里**找到。
  块越多、拼起来越长，命中当然越容易——所以**不同 block 预算之间不能直接比**。
  真正能比的只有同样块数下的两行，比如 topk=24 的 bm25 和 topk=24 的 panorama。
"""
import argparse
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (标签, mode, kinds, years, topk)；同 topk 的几行互相可比
CASES = [
    ("无元数据过滤",            "bm25", "none", "none", 8),
    ("+ 报告期过滤",            "bm25", "auto", "none", 8),
    ("+ 年份过滤",              "bm25", "auto", "auto", 8),
    ("年份过滤 + 召回 24",       "bm25", "auto", "auto", 24),
    ("年份过滤 + 召回 48",       "bm25", "auto", "auto", 48),
    ("年份过滤 + 全景扇出 24",    "panorama", "auto", "auto", 24),
]
VEC_CASES = [
    ("混合检索 mix，召回 8",     "mix", "auto", "auto", 8),
    ("混合检索 mix，召回 24",    "mix", "auto", "auto", 24),
    ("混合检索 mix，召回 48",    "mix", "auto", "auto", 48),
    ("纯向量 vec，召回 8",       "vec", "auto", "auto", 8),
    ("纯向量 vec，召回 24",      "vec", "auto", "auto", 24),
]


def run(mode, kinds, years, topk):
    """跑一档，返回 (数值覆盖字符串, 覆盖率)。"""
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "eval_retrieval.py"),
         "--modes", mode, "--kinds", kinds, "--years", years, "--topk", str(topk)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    if r.returncode != 0:
        return None, None
    for line in r.stdout.splitlines():
        if "全库数值覆盖" in line:
            s = line.split("全库数值覆盖")[1].strip().split()[0]
            try:
                return s, float(s.split("=")[-1].strip().rstrip("%"))
            except ValueError:
                return s, None
    return None, None          # 没有 REQ 的题跑不出来（比如向量缺失）

def width(s):
    """中文按两个字符宽算，否则表头对不齐（终端里中文占两格，format 当成一格）。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)


def pad(s, n, right=False):
    gap = " " * max(0, n - width(s))
    return (gap + s) if right else (s + gap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-vec", action="store_true",
                    help="连 mix/vec 一起跑（需要 data/index/vectors.npy）")
    args = ap.parse_args()

    vec_ready = os.path.exists(os.path.join(ROOT, "data", "index", "vectors.npy"))
    if args.with_vec and not vec_ready:
        print("！ vectors.npy 还没有，mix/vec 那几档会被跳过\n")

    print(pad("配置", 28) + pad("mode", 10) + pad("kinds", 7) + pad("years", 7)
          + pad("topk", 5, True) + "  " + pad("数值覆盖", 10, True))
    print("-" * 78)
    for label, mode, kinds, years, topk in CASES + (VEC_CASES if args.with_vec else []):
        # mix/vec 没向量什么都召不回；panorama 的底是 mix，向量缺失时自动退化成 BM25，
        # 所以它在建向量之前也跑得出数
        if mode in ("mix", "vec") and not vec_ready:
            row = "（跳过：无向量）"
        else:
            cov, _ = run(mode, kinds, years, topk)
            row = cov or "（无数据）"
        print(pad(label, 28) + pad(mode, 10) + pad(kinds, 7) + pad(years, 7)
              + pad(str(topk), 5, True) + "  " + pad(row, 10, True))

    print("\n注意：不同 topk 之间不可直接比——块越多拼起来越长，命中越容易。")
    print("同一 topk 下的两行才是有效对照（例如 24 的 bm25 vs 24 的 panorama）。")


if __name__ == "__main__":
    main()
