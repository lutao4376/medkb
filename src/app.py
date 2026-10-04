# -*- coding: utf-8 -*-
"""
问答页面。标准库 http.server，不引额外依赖。

  GET  /              页面
  GET  /api/stats     索引规模、公司清单
  POST /api/ask       流式回答（SSE）

回答里的 [n] 在页面上是可点的，点了滚到下面第 n 张证据卡；证据卡上带
公司/报告期/章节/页码/召回名次/来源（bm25 / vec / both）——作业要的"逐题溯源"
直接在这上面截图就行。
"""
import argparse
import json
import os
import socket
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa import SYSTEM, build_context, load_key  # noqa: E402
from search import Searcher  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

API = "https://api.deepseek.com/chat/completions"

STATE = {"searcher": None, "key": None, "model": "deepseek-v4-pro", "warm": False}
LOCK = threading.Lock()

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>医疗器械财报问答</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{--fg:#1a1d21;--mut:#6b7280;--line:#e5e7eb;--acc:#1d4ed8;--bg:#f7f8fa;
        --bm:#0e7490;--ve:#7c3aed;--bo:#b45309;--card:#fff}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
       color:var(--fg);background:var(--bg)}
  header{background:#fff;border-bottom:1px solid var(--line);padding:14px 24px;
         display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  h1{font-size:17px;margin:0;font-weight:650}
  .sub{color:var(--mut);font-size:13px}
  main{max-width:1080px;margin:0 auto;padding:20px 24px 80px}
  .bar{display:flex;gap:10px;margin-bottom:8px}
  input[type=text]{flex:1;padding:11px 14px;border:1px solid var(--line);border-radius:9px;
       font-size:15px;font-family:inherit;background:#fff}
  input[type=text]:focus{outline:none;border-color:var(--acc);box-shadow:0 0 0 3px #1d4ed81a}
  button{padding:11px 22px;border:0;border-radius:9px;background:var(--acc);color:#fff;
       font-size:15px;font-family:inherit;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  .opts{display:flex;gap:16px;align-items:center;color:var(--mut);font-size:13px;
        margin-bottom:18px;flex-wrap:wrap}
  .opts select,.opts input[type=number]{padding:4px 7px;border:1px solid var(--line);
        border-radius:6px;font-family:inherit;font-size:13px;background:#fff}
  .ex{color:var(--acc);cursor:pointer;margin-right:10px;font-size:12.5px}
  .ex:hover{text-decoration:underline}
  .card{background:var(--card);border:1px solid var(--line);border-radius:11px;
        padding:18px 20px;margin-bottom:14px}
  .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);
        margin:0 0 10px;font-weight:600}
  #ans{white-space:pre-wrap;line-height:1.8;min-height:24px}
  #ans .cit{color:var(--acc);font-weight:600;cursor:pointer;padding:0 1px;
        text-decoration:none;border-bottom:1px solid #1d4ed855}
  #ans .cit:hover{background:#1d4ed814}
  .meta{color:var(--mut);font-size:12.5px;margin-top:14px;padding-top:11px;
        border-top:1px dashed var(--line)}
  .ev{background:var(--card);border:1px solid var(--line);border-radius:11px;
      padding:14px 17px;margin-bottom:10px;scroll-margin-top:14px}
  .ev:target,.ev.flash{border-color:var(--acc);box-shadow:0 0 0 3px #1d4ed81f}
  .ev .hd{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:8px;
      font-size:12.5px;color:var(--mut)}
  .num{display:inline-flex;align-items:center;justify-content:center;min-width:23px;height:23px;
      padding:0 6px;border-radius:6px;background:#eef2ff;color:var(--acc);font-weight:700;
      font-size:12.5px}
  .tag{padding:1.5px 7px;border-radius:5px;font-size:11.5px;font-weight:600;color:#fff}
  .t-both{background:var(--bo)}.t-bm25{background:var(--bm)}.t-vec{background:var(--ve)}
  .t-tab{background:#475569}.t-txt{background:#94a3b8}
  .ev .body{white-space:pre-wrap;font-size:13.5px;line-height:1.62;
      max-height:230px;overflow:auto;background:#fbfbfd;border-radius:7px;padding:10px 12px;
      font-family:ui-monospace,Consolas,"Microsoft YaHei",monospace}
  .ev .body.mono{font-size:12.5px}
  .co{color:var(--fg);font-weight:650}
  .hint{color:var(--mut);font-size:13px;line-height:1.9}
  .filters{font-size:12.5px;color:var(--acc);background:#eef2ff;border:1px solid #dbe3fb;
      border-radius:8px;padding:6px 11px;margin-bottom:12px}
  .filters:empty{display:none}
  .spin{display:inline-block;width:13px;height:13px;border:2px solid #c7d2fe;
      border-top-color:var(--acc);border-radius:50%;animation:sp .7s linear infinite;
      vertical-align:-2px;margin-right:7px}
  @keyframes sp{to{transform:rotate(360deg)}}
</style></head><body>
<header>
  <h1>医疗器械上市公司财报问答</h1>
  <span class="sub" id="stats">加载中…</span>
</header>
<main>
  <div class="bar">
    <input type="text" id="q" placeholder="例：迈瑞医疗 2025 年营业收入是多少？"
           onkeydown="if(event.key==='Enter')go()">
    <button id="btn" onclick="go()">提问</button>
  </div>
  <div class="opts">
    <label>检索方式
      <select id="mode">
        <option value="mix">向量 + BM25 混合</option>
        <option value="vec">只用向量</option>
        <option value="bm25">只用 BM25</option>
        <option value="panorama">全景扇出（每家各取）</option>
      </select></label>
    <label>召回 <input type="number" id="topk" value="8" min="1" max="40" style="width:52px"> 块</label>
    <label>报告期
      <select id="kinds">
        <option value="auto">自动（按题面推）</option>
        <option value="年度">只看年度报告</option>
        <option value="半年度">只看半年度报告</option>
        <option value="all">不限</option>
      </select></label>
    <label><input type="checkbox" id="llm" checked> 调模型作答</label>
    <span class="sub" style="margin-left:auto">示例：</span>
    <span id="exs"></span>
  </div>

  <div class="card" id="anscard" style="display:none">
    <h2>回答</h2><div id="ans"></div><div class="meta" id="ameta"></div>
  </div>
  <div class="filters" id="filters"></div>
  <div id="evs"></div>
  <div class="hint" id="hint">
    问题会先在两套索引里各召回 50 块（向量 + BM25），融合后限制同一家公司最多 3 块，
    再交给模型只依据召回到的材料作答；答案里每个 <b>[n]</b> 都能点开看原文。
  </div>
</main>
<script>
const $=s=>document.querySelector(s);
let EVS=[];

fetch('/api/stats').then(r=>r.json()).then(d=>{
  $('#stats').textContent=`${d.chunks} 个文本块 · ${d.companies} 家上市公司 · ${d.reports} 份报告`
    +` · 向量 ${d.has_vec?'已建':'未建'} · BM25 索引字段 ${d.bm25_field}`;
  $('#exs').innerHTML=d.samples.map((s,i)=>
      `<span class="ex" onclick="fill(${i})">${s}</span>`).join('');
  window.SAMPLES=d.samples;
});
function fill(i){$('#q').value=SAMPLES[i];go();}

function anchor(n){return `<span class="cit" onclick="jump(${n})">[${n}]</span>`;}
function render(t){
  const esc=t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  $('#ans').innerHTML=esc.replace(/\[(\d+)\]/g,(m,n)=>anchor(+n));
}
function jump(n){
  const el=document.getElementById('ev'+n); if(!el)return;
  el.scrollIntoView({block:'start',behavior:'smooth'});
  el.classList.add('flash'); setTimeout(()=>el.classList.remove('flash'),1400);
}
function tagSrc(s){return `<span class="tag t-${s}">${s==='both'?'两路命中':s==='bm25'?'仅 BM25':'仅向量'}</span>`;}

function go(){
  const q=$('#q').value.trim(); if(!q)return;
  const btn=$('#btn'); btn.disabled=true;
  // 截图脚本靠这个标志判断"答完了"——SSE 流没结束就截图只会拍到"检索中…"
  window.__qa={state:'running',q};
  $('#anscard').style.display='block';
  $('#evs').innerHTML=''; $('#hint').style.display='none';
  $('#ameta').textContent='';
  const useLlm=$('#llm').checked;
  $('#ans').innerHTML=useLlm?'<span class="spin"></span>检索中…':'<span class="spin"></span>检索中…';

  fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({q, mode:$('#mode').value, topk:+$('#topk').value,
                         kinds:$('#kinds').value, llm:useLlm})})
  .then(async r=>{
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='', raw='';
    while(true){
      const {done,value}=await rd.read(); if(done)break;
      buf+=dec.decode(value,{stream:true});
      let i;
      while((i=buf.indexOf('\n\n'))>=0){
        const chunk=buf.slice(0,i); buf=buf.slice(i+2);
        if(!chunk.startsWith('data: '))continue;
        const ev=JSON.parse(chunk.slice(6));
        if(ev.type==='hits'){EVS=ev.hits; drawHits(ev.hits); $('#ans').innerHTML='';
          const f=[];
          if(ev.only&&ev.only.length)f.push('定向公司 '+ev.only.join('、'));
          if(ev.kinds&&ev.kinds.length)f.push('报告期 '+ev.kinds.join('/'));
          if(ev.years&&ev.years.length)f.push('年份 '+ev.years.join('/'));
          if(ev.panorama)f.push('全景扇出：每家各取，'+ev.hits.length+' 块');
          $('#filters').innerHTML=f.length?('已应用：'+f.join(' · ')):'';}
        else if(ev.type==='delta'){raw+=ev.text; render(raw);}
        else if(ev.type==='done'){
          if(!raw)render(ev.answer||'');
          $('#ameta').innerHTML=ev.meta||'';
        }
        else if(ev.type==='error'){window.__qa={state:'error',msg:ev.message};
                                   render('出错：'+ev.message);}
      }
    }
    btn.disabled=false;
    if(window.__qa&&window.__qa.state==='running')window.__qa={state:'done'};
  })
  .catch(e=>{$('#ans').innerHTML='请求失败：'+e; btn.disabled=false;
             window.__qa={state:'error',msg:String(e)};});
}

// 深链：/ ?q=…&mode=mix&topk=8&kinds=年度&llm=0&auto=1
// 无头浏览器没法点按钮，截图只能靠这个参数把问题喂进来。
// 顺带也方便把某个具体问题发给别人复现。
(function(){
  const p=new URLSearchParams(location.search);
  if(p.get('mode'))$('#mode').value=p.get('mode');
  if(p.get('topk'))$('#topk').value=p.get('topk');
  if(p.get('kinds'))$('#kinds').value=p.get('kinds');
  if(p.get('llm')==='0')$('#llm').checked=false;
  const q=p.get('q');
  if(q){$('#q').value=q; go();}
})();

function drawHits(hits){
  $('#evs').innerHTML=hits.map((h,i)=>{
    const n=i+1;
    const rows=[];
    if(h._rank_bm25)rows.push('BM25 #'+h._rank_bm25);
    if(h._rank_vec)rows.push('向量 #'+h._rank_vec);
    return `<div class="ev" id="ev${n}">
      <div class="hd">
        <span class="num">${n}</span>
        <span class="co">${h.name}</span>
        <span>${h.year}年${h.kind}报告</span>
        <span>第 ${h.page} 页</span>
        <span>${h.section||'-'}</span>
        ${tagSrc(h._source)}
        <span class="tag ${h.ctype==='table'?'t-tab':'t-txt'}">${h.ctype==='table'?'表格':'正文'}</span>
        <span style="margin-left:auto">${rows.join(' · ')}</span>
      </div>
      <div class="body ${h.ctype==='table'?'mono':''}">${h.text.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</div>
    </div>`;}).join('');
}
</script></body></html>
"""


class Server(ThreadingHTTPServer):
    """
    浏览器关标签页、或中途放弃生成时，会把正在流式的 SSE 连接直接掐断。
    socketserver 把这当异常，默认打一整段 traceback——可这是正常用法，不是故障
    （`logs/app_run.err` 里刷了十几段）。控制台一片红字会让人以为页面坏了，
    所以只吞这一类和写回失败，别的照旧抛。

    handle_error 在 **server** 上，不在 handler 上——所以只给 Handler 加
    log_message 静音是拦不住的，那段 traceback 照样出来。
    """

    daemon_threads = True

    def handle_error(self, request, client_address):
        if sys.exc_info()[0] in (ConnectionResetError, ConnectionAbortedError,
                                 BrokenPipeError):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass  # 别刷屏

    def _send(self, code, ctype, body, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            return self._send(200, "text/html; charset=utf-8", PAGE)
        if p == "/api/stats":
            s = STATE["searcher"]
            reports = {(c["code"], c["year"], c["kind"]) for c in s.chunks}
            return self._send(200, "application/json; charset=utf-8", json.dumps({
                "chunks": len(s.chunks),
                "companies": len(s.companies),
                "reports": len(reports),
                "has_vec": s.vectors is not None,
                "bm25_field": s.bm25_field,
                "samples": [
                    "迈瑞医疗 2025 年的营业收入、归母净利润、扣非归母净利润分别是多少？",
                    "2025 年度营业收入最高的 6 家公司分别是哪几家？各是多少？",
                    "2025 年度营业收入同比下滑的有哪几家公司？谁的降幅最大？",
                    "哪些公司的 2025 年报里提到了集采（带量采购）？它们分别是怎么描述的？",
                ],
            }, ensure_ascii=False))
        self._send(404, "text/plain", "not found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            return self._send(404, "text/plain", "not found")
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        q = (req.get("q") or "").strip()
        mode = req.get("mode", "mix")
        kinds = req.get("kinds", "auto")
        if kinds == "all":
            kinds = None
        topk = int(req.get("topk", 8))
        use_llm = bool(req.get("llm", True))
        s = STATE["searcher"]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj):
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        try:
            hits = s.search(q, topk=topk, mode=mode, per_company=3, trace=True,
                            kinds=kinds)
            emit({"type": "hits", "hits": hits,
                  "kinds": s.last_kinds, "years": s.last_years,
                  "panorama": s.last_panorama, "only": s.last_only})

            if not use_llm:
                emit({"type": "done", "answer": "（已关闭模型作答，仅展示检索结果）",
                      "meta": f"召回 {len(hits)} 块 · 检索方式 {mode}"})
                return
            if not hits:
                emit({"type": "done", "answer": "未检索到相关材料。", "meta": ""})
                return

            key = STATE["key"]
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                    f"【材料】\n{build_context(hits)}\n\n【问题】\n{q}\n\n请依据以上材料回答，逐处标注编号。"},
            ]
            import requests
            r = requests.post(API, timeout=180, stream=True,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": STATE["model"], "messages": msgs,
                      "temperature": 0.0, "stream": True})
            r.raise_for_status()
            acc = ""
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                t = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
                if t:
                    acc += t
                    emit({"type": "delta", "text": t})

            import re
            cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", acc)})
            bad = [c for c in cited if not 1 <= c <= len(hits)]
            flt = []
            if s.last_only:
                flt.append("定向 " + "、".join(s.last_only))
            if s.last_kinds:
                flt.append("报告期 " + "/".join(s.last_kinds))
            meta = (f"引用 {cited or '无'} · 共 {len(hits)} 块 · {mode} · "
                    + ("；".join(flt) if flt else "无过滤")
                    + (f" · <b style='color:#b91c1c'>越界编号 {bad}</b>" if bad else " · 编号均有效"))
            emit({"type": "done", "answer": acc, "meta": meta})
        except Exception as e:
            traceback.print_exc()
            try:
                emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    def do_OPTIONS(self):
        self._send(204, "text/plain", b"")


def warm(s):
    """后台把向量模型读进内存，省得第一次提问等十几秒。"""
    s._encoder()
    STATE["warm"] = True
    print("[warm] 向量模型就绪", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--index", default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()

    # 先看端口上是不是已经有人了。Windows 的 SO_REUSEADDR 语义和 Unix 不一样：
    # 它允许**抢占**已绑定的端口，第二个实例照样 bind 成功、照样打印"已启动"，
    # 但连接全被前一个进程接走——表现出来就是"改了代码没生效"，
    # 而且两个进程都 LISTENING，不 netstat 根本看不出来。所以宁可起不来。
    probe = socket.socket()
    probe.settimeout(0.6)
    if probe.connect_ex(("127.0.0.1", args.port)) == 0:
        probe.close()
        sys.exit(f"端口 {args.port} 上已经有东西在服务了。先把它停掉"
                 f"（netstat -ano | findstr :{args.port} 拿 PID），"
                 f"或者换个 --port。刚才那个进程跑的是旧代码。")
    probe.close()

    from search import IDX
    STATE["searcher"] = Searcher(args.index or IDX)
    STATE["model"] = args.model
    if not args.no_llm:
        STATE["key"] = load_key()

    threading.Thread(target=warm, args=(STATE["searcher"],), daemon=True).start()

    srv = Server(("127.0.0.1", args.port), Handler)
    print(f"http://127.0.0.1:{args.port}   "
          f"(索引 {len(STATE['searcher'].chunks)} 块，Ctrl+C 退出)", file=sys.stderr)
    srv.serve_forever()


if __name__ == "__main__":
    main()
