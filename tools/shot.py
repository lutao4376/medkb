# -*- coding: utf-8 -*-
"""
给问答页面截图。

不能用 `chrome --screenshot`：页面是 SSE 流式的，模型要流十几秒，而
--virtual-time-budget 不等 fetch 的流式 body，截出来只有一句"检索中…"。
所以走 CDP：开一个带调试端口的无头 Chrome，导航过去，轮询页面上的
window.__qa 标志（app.py 里 go() 写的），等它变成 done 再截图。

用法:
    python tools/shot.py --shot 01-单公司 --q "迈瑞医疗 2025 年的营业收入是多少？"
    python tools/shot.py --shot 02-全景 --q "…" --mode mix --kinds 年度
    python tools/shot.py --shot 00-空白          # 不提问，截初始界面
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--q", default=None)
    ap.add_argument("--mode", default="mix")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--kinds", default="auto")
    ap.add_argument("--llm", default="1")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--max-px", type=int, default=0,
                    help="页面逻辑高度上限；超过了就裁到这么高。"
                         "召回 24 块时整页有 7000+ 逻辑像素，缩进交付文档里字就看不清了")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--dbg-port", type=int, default=9222)
    args = ap.parse_args()

    import websocket                                   # websocket-client

    exe = next((p for p in BROWSERS if os.path.exists(p)), None)
    if not exe:
        raise SystemExit("找不到 Chrome/Edge")

    # 页面先活着，否则截到的是 ERR_CONNECTION_REFUSED
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/stats", timeout=5) as r:
            st = json.load(r)
    except Exception as e:
        raise SystemExit(f"页面没起（{e}）：先跑 python src/app.py --port {args.port}")
    print(f"页面在线：{st['chunks']} 块 · 向量 {'已建' if st.get('has_vec') else '未建'}"
          f" · BM25 字段 {st.get('bm25_field')}")

    prof = tempfile.mkdtemp(prefix="shot-")
    proc = subprocess.Popen(
        [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--no-first-run", "--no-default-browser-check",
         f"--remote-debugging-port={args.dbg_port}", f"--user-data-dir={prof}",
         f"--window-size={args.width},{args.height}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ws = None
    try:
        target = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.dbg_port}/json/list", timeout=2) as r:
                    tabs = [t for t in json.load(r) if t.get("type") == "page"]
                if tabs:
                    target = tabs[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not target:
            raise SystemExit("Chrome 调试端口没起来")

        # 新版 Chrome 会按 Origin 拒掉 WS 握手；发请求时干脆不带 Origin
        ws = websocket.create_connection(target, timeout=args.timeout, suppress_origin=True)
        mid = [0]

        def cmd(method, params=None):
            mid[0] += 1
            ws.send(json.dumps({"id": mid[0], "method": method, "params": params or {}}))
            while True:
                m = json.loads(ws.recv())
                if m.get("id") == mid[0]:
                    if "error" in m:
                        raise RuntimeError(f"{method}: {m['error']}")
                    return m.get("result", {})

        def js(expr):
            r = cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
            return r.get("result", {}).get("value")

        cmd("Page.enable")
        cmd("Runtime.enable")
        cmd("Emulation.setDeviceMetricsOverride",
            {"width": args.width, "height": args.height,
             "deviceScaleFactor": args.scale, "mobile": False})

        params = {"mode": args.mode, "topk": args.topk,
                  "kinds": args.kinds, "llm": args.llm}
        if args.q:
            params["q"] = args.q
        url = f"http://127.0.0.1:{args.port}/?" + urllib.parse.urlencode(params)
        cmd("Page.navigate", {"url": url})

        if args.q:
            t0, last = time.time(), None
            while time.time() - t0 < args.timeout:
                st_ = js("window.__qa && window.__qa.state")
                if st_ and st_ != "running":
                    last = st_
                    break
                if st_ and st_ != last:
                    print(f"  [{time.time()-t0:5.1f}s] {st_}")
                    last = st_
                time.sleep(0.5)
            else:
                print("  ⚠ 超时，按当前状态截")
            print(f"  [{time.time()-t0:5.1f}s] 结束状态 {last}")
            if last == "error":
                print("  页面报错：", js("window.__qa && window.__qa.msg"))
            time.sleep(0.8)                            # 等最后一段 DOM 落地
        else:
            time.sleep(2.5)

        # 先量一下完整高度，再按整页截
        h = js("Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)") or args.height
        h = int(min(h + 30, args.max_px or 6000))
        cmd("Emulation.setDeviceMetricsOverride",
            {"width": args.width, "height": h,
             "deviceScaleFactor": args.scale, "mobile": False})
        time.sleep(0.4)
        # clip 里的 scale 是**再乘一遍** deviceScaleFactor 的，用 max-px 裁的时候
        # 必须把 deviceScaleFactor 压回 1，否则图会变成 4 倍像素、文件大到没法用
        params = {"format": "png", "captureBeyondViewport": True}
        if args.max_px:
            params["clip"] = {"x": 0, "y": 0, "width": args.width, "height": h,
                              "scale": args.scale}
            cmd("Emulation.setDeviceMetricsOverride",
                {"width": args.width, "height": h,
                 "deviceScaleFactor": 1, "mobile": False})
            time.sleep(0.3)
        shot = cmd("Page.captureScreenshot", params)

        out_dir = os.path.join(ROOT, "docs", "shots")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, args.shot + ".png")
        with open(out, "wb") as f:
            f.write(base64.b64decode(shot["data"]))
        print(f"-> {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    finally:
        try:
            if ws:
                ws.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
