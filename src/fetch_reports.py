# -*- coding: utf-8 -*-
"""
从巨潮资讯网（cninfo，交易所官方指定披露网站）抓取年报/半年报全文 PDF。
巨潮同时覆盖沪深两市，一条管线即可。

用法:
    python fetch_reports.py --list           # 只列公告，不下载
    python fetch_reports.py --download       # 列出并下载
    python fetch_reports.py --download --only 300760
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:  # Windows 控制台默认 GBK，会把中文输出打成乱码
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "data", "pdf")
META_PATH = os.path.join(ROOT, "data", "reports_meta.json")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
H = {"User-Agent": UA, "Referer": "http://www.cninfo.com.cn/"}

# ---------------------------------------------------------------- 公司名单
# 宽口径混合 12 家：影像设备 / 体外诊断 / 高值耗材介入 / 骨科 / 眼科 / 内镜 / 家用
COMPANIES = [
    ("300760", "迈瑞医疗", "综合平台（生命信息与支持 / IVD / 医学影像）"),
    ("688271", "联影医疗", "高端医学影像设备"),
    ("603658", "安图生物", "体外诊断 IVD（化学发光 / 微生物）"),
    ("300633", "开立医疗", "超声诊断 + 软性内镜"),
    ("688212", "澳华内镜", "软性内镜（消化内镜）"),
    ("688029", "南微医学", "内镜诊疗器械"),
    ("300003", "乐普医疗", "心血管介入（支架 / 器械 / 药品）"),
    ("688617", "惠泰医疗", "电生理 + 血管介入"),
    ("688016", "心脉医疗", "主动脉及外周血管介入"),
    ("002901", "大博医疗", "骨科植入耗材"),
    ("688050", "爱博医疗", "眼科（人工晶状体 / 角膜塑形镜）"),
    ("002223", "鱼跃医疗", "家用医疗器械"),
]

# 近两年 = 2 份年报 + 2 份半年报
KEEP_YEARS = {"2024", "2025", "2026"}
KEEP_KINDS = {"年度", "半年度"}
# 只要这些年份的半年报（2024 年的半年报不在"近两年"窗口内）
KEEP = {("2024", "年度"), ("2025", "年度"), ("2025", "半年度"), ("2026", "半年度")}

DROP_PAT = re.compile(r"摘要|英文|English|EN\b|更正|取消|补充|已取消|问询|回复|说明|公告$")


def market_of(code: str):
    """6 开头 -> 沪市；0/3 开头 -> 深市"""
    return ("sse", "sh") if code.startswith("6") else ("szse", "sz")


def get_org_id(code: str) -> str | None:
    r = requests.post(
        "http://www.cninfo.com.cn/new/information/topSearch/query",
        data={"keyWord": code, "maxNum": 10},
        headers={**H, "Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    r.raise_for_status()
    for item in r.json():
        if item.get("code") == code:
            return item.get("orgId")
    return None


def list_reports(code: str, org_id: str, debug: bool = False):
    col, plate = market_of(code)
    r = requests.post(
        "http://www.cninfo.com.cn/new/hisAnnouncement/query",
        data={
            "pageNum": 1,
            "pageSize": 60,
            "column": col,
            "tabName": "fulltext",
            "plate": plate,
            "stock": f"{code},{org_id}",
            "searchkey": "",
            "secid": "",
            "category": "category_ndbg_szsh;category_bndbg_szsh",
            "trade": "",
            "seDate": "",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        },
        headers={**H, "Content-Type": "application/x-www-form-urlencoded",
                 "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice"},
        timeout=25,
    )
    r.raise_for_status()
    items = r.json().get("announcements") or []
    if debug:
        for a in items:
            print("   [all]", a["announcementTitle"])
    return items


def pick(items, debug=False):
    """从公告列表里挑出我们要的 4 份全文"""
    out = {}
    for a in items:
        title = a["announcementTitle"]
        m = re.search(r"(20\d{2})\s*年\s*(年度|半年度)\s*报告\s*$", title)
        if not m:
            continue
        year, kind = m.group(1), m.group(2)
        if (year, kind) not in KEEP:
            continue
        tail = title[m.start():]
        if DROP_PAT.search(title.replace(tail, "")) or DROP_PAT.search(title[: m.start()]):
            if debug:
                print("   [drop]", title)
            continue
        key = f"{year}{kind}"
        if key in out:
            if debug:
                print("   [dup ]", title)
            continue
        out[key] = {
            "code": None,
            "title": title,
            "url": "http://static.cninfo.com.cn/" + a["adjunctUrl"],
            "date": a.get("announcementTime"),
            "size": a.get("adjunctSize"),
            "year": year,
            "kind": kind,
        }
    return out


def download_one(rec, dest, retries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
        return "skip", os.path.getsize(dest)
    for i in range(retries):
        try:
            r = requests.get(rec["url"], headers=H, timeout=120, stream=True)
            r.raise_for_status()
            tmp = dest + ".part"
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
                    n += len(chunk)
            os.replace(tmp, dest)
            return "ok", n
        except Exception as e:
            if i == retries - 1:
                return f"fail:{e}", 0
            time.sleep(2 * (i + 1))
    return "fail", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--only", default=None, help="只处理某个代码")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(PDF_DIR, exist_ok=True)
    all_meta = {}
    tasks = []

    for code, name, sector in COMPANIES:
        if args.only and code != args.only:
            continue
        try:
            org = get_org_id(code)
            if not org:
                print(f"[!] {code} {name}: 查不到 orgId")
                continue
            items = list_reports(code, org, args.debug)
            got = pick(items, args.debug)
            print(f"[{code}] {name}  orgId={org}  公告{len(items)}条 -> 选中{len(got)}份: "
                  + ", ".join(sorted(got)))
            for k, v in got.items():
                v["code"] = code
                v["name"] = name
                v["sector"] = sector
                dest = os.path.join(PDF_DIR, f"{code}_{name}_{v['year']}{v['kind']}报告.pdf")
                v["path"] = dest
                if args.download:
                    tasks.append((v, dest))
            all_meta[code] = {"name": name, "sector": sector, "orgId": org, "reports": got}
        except Exception as e:
            print(f"[!] {code} {name}: {e}")

    if args.download and tasks:
        print(f"\n开始下载 {len(tasks)} 份 PDF ...")
        t0 = time.time()
        total = 0
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(download_one, rec, dest): (rec, dest) for rec, dest in tasks}
            for i, fu in enumerate(as_completed(futs), 1):
                rec, dest = futs[fu]
                st, n = fu.result()
                total += n
                print(f"  [{i}/{len(tasks)}] {st:12s} {n/1e6:7.2f}MB  {os.path.basename(dest)}")
        print(f"完成：{total/1e6:.1f}MB，用时 {time.time()-t0:.1f}s")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(f"\n元数据 -> {META_PATH}")


if __name__ == "__main__":
    main()
