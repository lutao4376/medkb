# -*- coding: utf-8 -*-
"""
向量索引的实时进度条。

读 data/index/progress.json（编码器每批覆盖一次），每 2 秒重画一行。
用法：
    python src/watch.py            # 一直刷到跑完
    python src/watch.py --once     # 只打印一次（给脚本/汇报用）
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROG = os.path.join(ROOT, "data", "index", "progress.json")
LOG = os.path.join(ROOT, "logs", "vec_build.log")

BAR = 34


def hhmmss(s):
    s = int(max(0, s))
    if s >= 3600:
        return f"{s//3600}小时{s%3600//60}分"
    if s >= 60:
        return f"{s//60}分{s%60}秒"
    return f"{s}秒"


def bar(pct, width=BAR):
    done = int(round(pct * width))
    return "█" * done + "░" * (width - done)


def render(prev=None):
    """
    进度按 token 算，不按块数。同样是 1 块，短的 180 token、长的 900 token，
    按块数算出来的 ETA 会乐观一倍（实测过：块口径 42 分，token 口径 100 分）。
    """
    if not os.path.exists(PROG):
        return None, "等待编码器写出第一个进度…", prev
    try:
        d = json.load(open(PROG, encoding="utf-8"))
    except Exception:
        return None, "读取进度文件…", prev

    done, total = d["done"], d["total"]
    dtok = d.get("done_tok") or 0
    ttot = d.get("total_tok") or 0
    pct = (dtok / ttot) if ttot else (done / total if total else 0)

    now = time.time()
    trate = d.get("rate_tok", 0) or 0
    if prev and prev[2] and dtok:
        dt, di = now - prev[0], dtok - prev[2]
        if dt > 0.5 and di > 0:
            inst = di / dt
            trate = inst if trate <= 0 else 0.75 * trate + 0.25 * inst
    eta = (ttot - dtok) / trate if trate > 0 else (d.get("eta") or 0)

    stale = now - d.get("ts", now)
    tokline = (f"{dtok/1e6:.2f}M / {ttot/1e6:.2f}M token   " if ttot else "")
    line = (
        f"  {bar(pct)} {pct*100:5.1f}%\n"
        f"  {tokline}{done:,} / {total:,} 块   第 {d['shard']+1}/{d['n_shard']} 片\n"
        f"  速率 {trate:,.0f} token/秒 ({d.get('rate_item',0):4.1f} 块/秒)   "
        f"本片已跑 {hhmmss(d['elapsed'])}   剩余约 {hhmmss(eta)}\n"
        f"  预计完成 {datetime.fromtimestamp(now+eta):%H:%M:%S}"
        + (f"    ⚠ 进度文件 {stale:.0f} 秒未更新" if stale > 90 else "")
    )
    return (now, done, dtok), line, (now, done, dtok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    prev = None
    if args.once:
        _, line, _ = render(prev)
        print(line)
        return
    try:
        while True:
            prev, line, _ = render(prev)
            w = shutil.get_terminal_size((80, 20)).columns
            out = "\n".join(l[:w] for l in (line or "").split("\n"))
            sys.stdout.write("\033[H\033[J" + out + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
