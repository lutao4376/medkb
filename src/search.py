# -*- coding: utf-8 -*-
"""
检索层：BM25、向量、以及两者的 RRF 融合。

两边的分数量纲完全不同（BM25 是无上界的数，向量是余弦），直接加权没有意义，
所以用 RRF（Reciprocal Rank Fusion）——只看名次，不看分数：
    score(d) = Σ_r 1 / (k + rank_r(d))
这是混合检索里最稳的做法，不用调权重。
"""
import json
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from index import BM25, tokenize, MODEL  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "data", "index")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"

RRF_K = 60
# 检索任务的指令前缀（Qwen3-Embedding 官方推荐的 query 侧写法，文档侧裸文本）
TASK = "给定用户关于医疗器械上市公司财报的问题，检索最有用的段落"

# 报告期识别。语料里有 年度 和 半年度 两种报告，同名章节（都有"主要会计数据"）
# 靠词面分不开——实测问"2025年营业收入"会捞到 2025 半年报，模型还会据此
# 断言"全年未披露"。所以从题面推报告期，先做一层元数据过滤。
RE_HALF = re.compile(r"半年|中报")
RE_ANN = re.compile(r"年报|年度|全年|年末|年报表")
RE_YEAR = re.compile(r"20\d{2}\s*年")
RE_YEARS = re.compile(r"(20\d{2})\s*年")


def infer_kinds(q):
    """从题面推断该查哪种报告。返回 None 表示不限。"""
    if RE_HALF.search(q):
        return ["半年度"]
    if RE_ANN.search(q):
        return ["年度"]
    if RE_YEAR.search(q):
        return ["年度"]          # 只说年份、没说是半年报，默认要全年数
    return None


def infer_years(q):
    """
    题面点名的年份。只提一个年份时才拿它过滤。

    光按 kind 过滤是不够的：语料里同时有 2024 年报和 2025 年报，两者 kind 都是
    "年度"，所以问"2025 年营业收入"照样会捞到 2024 年报的块当 2025 的数用。
    扇出检索把这一点放大了——逐家取前 1 块时，南微回来的就是 2024 年报。

    提了两个年份（"2024 和 2025 各是多少"）就不过滤，那种题要的正是两年。
    """
    ys = sorted(set(RE_YEARS.findall(q)))
    return ys if len(ys) == 1 else None


class Searcher:
    def __init__(self, index_dir=IDX, need_vec=True, q_device="cpu"):
        self.dir = index_dir
        self.q_device = q_device
        self.chunks = [json.loads(l) for l in
                       open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8")]

        # BM25 优先用正文口径（见 index.py 的 BM25_FIELD 注释），没有就退回旧口径
        p = None
        for f in ("bm25_text.pkl", "bm25.pkl"):
            if os.path.exists(os.path.join(index_dir, f)):
                p = os.path.join(index_dir, f)
                break
        if p is None:
            raise FileNotFoundError(f"{index_dir} 下没有 bm25_*.pkl，先跑 src/index.py")
        with open(p, "rb") as f:
            d = pickle.load(f)
        self.bm25_file = os.path.basename(p)
        self.bm25_field = d.get("field") or ("text" if "text" in self.bm25_file else "embed_text")
        self.bm25 = BM25([[]])            # 占位，下面直接灌入
        self.bm25.postings = d["postings"]
        self.bm25.idf = d["idf"]
        self.bm25.dl = d["dl"]
        self.bm25.avgdl = d["avgdl"]
        self.bm25.k1, self.bm25.b, self.bm25.N = d["k1"], d["b"], d["N"]

        self.vectors = None
        self.model = None
        if need_vec and os.path.exists(os.path.join(index_dir, "vectors.npy")):
            self.vectors = np.load(os.path.join(index_dir, "vectors.npy"))
            assert len(self.vectors) == len(self.chunks)

        # 各家公司名，用来做查询里的公司识别
        self.companies = sorted({c["name"] for c in self.chunks})
        # 最近一次 search() 实际施加的过滤，给页面和记录看
        self.last_kinds = None
        self.last_only = None
        self.last_panorama = False
        self.last_years = None
        # 全景题要拿同一个 query 逐家问 12 次。查询向量和 BM25 候选只跟题面有关、
        # 跟公司无关，缓存一次就够——不然 12 次里 11 次是白算的。
        self._vq_cache = None
        self._bq_cache = None

    # -------------------------------------------------- 两个单路检索
    def search_bm25(self, q, k=50):
        return self.bm25.search(tokenize(q), topk=k)

    def _bm25_cached(self, q, k):
        """同题同 k 的 BM25 候选缓存。分词很贵，全景题会把同一个 q 问 12 遍。"""
        key = (q, k)
        if self._bq_cache is None or self._bq_cache[0] != key:
            self._bq_cache = (key, self.search_bm25(q, k))
        return self._bq_cache[1]

    def _encoder(self):
        """
        查询侧必须和建索引时用同一个编码器——否则 padding/pooling 只要有一点不同，
        余弦分就整体漂移，召回的块会悄悄换人。这里默认走 CPU：
        一条 query 才几十个 token，CPU 也就几十毫秒，还能避免和建索引抢显存。
        """
        if self.model is None:
            from encoder import Encoder, MAXLEN
            self.model = Encoder(device=self.q_device, half=False)
            self.model.maxlen = MAXLEN
        return self.model

    def search_vec(self, q, k=50):
        if self.vectors is None:
            return []
        if self._vq_cache is None or self._vq_cache[0] != q:
            m = self._encoder()
            v = m.encode([f"Instruct: {TASK}\nQuery: {q}"],
                         maxlen=getattr(m, "maxlen", 1536))[0].astype(np.float32)
            self._vq_cache = (q, v)
        v = self._vq_cache[1]
        s = self.vectors @ v
        n = min(k, len(s))
        idx = np.argpartition(-s, n - 1)[:n]
        idx = idx[np.argsort(-s[idx])]
        return [(int(i), float(s[i])) for i in idx]

    # -------------------------------------------------- 融合
    def _fanout(self, q, topk, base, kinds, years, trace):
        """
        全景题扇出：逐家各取 per_each 块，再轮转合并。

        为什么非扇出不可：全景题问的是"12 家里谁最高、谁在下滑"，而单轮 top-k
        不管怎么排都只有 8 块——12 家的数塞不进 8 块里，模型只能"看了三家就下结论"。
        这不是排序排偏了，是候选池的容量问题，调分数、调权重都没用。
        逐家问一次，把"每家必须有"变成硬约束，是唯一在检索层能解决它的办法。

        合并用轮转（R1：每家的第一块；R2：每家的第二块……）而不是按分数排：
        按分数排又会退化成"迈瑞的 3 块排前面、后 4 家被截断"，等于没扇出。
        """
        if kinds == "auto":
            kinds = infer_kinds(q)
        if years == "auto":
            years = infer_years(q)
        n = len(self.companies)
        # topk 在这里的含义变成"总块数"，向上取整到每家整数块
        per_each = max(1, -(-topk // n))
        pools = []
        for name in self.companies:
            rs = self.search(q, topk=per_each, mode=base, only=[name],
                             kinds=kinds, years=years, trace=trace, panorama=False)
            for c in rs:
                c["_panorama"] = name
            pools.append(rs)
        out, r = [], 0
        while True:
            added = False
            for p in pools:
                if r < len(p):
                    out.append(p[r])
                    added = True
            if not added:
                break
            r += 1
        # 内层 12 次调用会把这三个状态踩成"最后一家公司"的，返回前按扇出重新写上，
        # 否则页面上显示成"只查了澳华内镜"，排查时会被带偏
        self.last_kinds = kinds
        self.last_years = years
        self.last_only = None
        self.last_panorama = True
        return out

    def search(self, q, topk=8, mode="mix", per_company=3, only=None, trace=False,
               kinds="auto", years="auto", panorama=None):
        """
        mode:  bm25 / vec / mix / panorama（panorama = 逐家扇出的 mix）
        kinds: "auto" 按题面推报告期；None 不限；也可直接传 ["年度"] / ["半年度"]
        years: "auto" 按题面推年份；None 不限；也可传 ["2025"]。
               只按 kind 过滤挡不住 2024 年报——它 kind 也是"年度"。

        先各取 50 条候选，RRF 融合；再按 per_company 限制同一家公司的块数，
        免得 10 条全来自迈瑞（全景题就没法比了）。

        trace=True 时每一条额外带 rank_bm25 / rank_vec / source——
        作业要"逐题记录召回了哪些块、卡在哪"，靠这个定位是 BM25 漏了还是向量漏了。
        """
        base = "mix" if mode == "panorama" else mode
        if panorama is None:
            panorama = (mode == "panorama")
        if kinds == "auto":
            kinds = infer_kinds(q)
            self.last_kinds = kinds
        if years == "auto":
            years = infer_years(q)
            self.last_years = years

        # 全景 = 题面没点名任何公司、却要求横向比较。
        # 一旦点名（"迈瑞的营收"），扇出就没意义了，退回普通检索。
        if panorama and only is None and not any(c in q for c in self.companies):
            return self._fanout(q, topk, base, kinds, years, trace)
        self.last_panorama = False

        # 公司名当过滤器，不当检索词。
        # 把它当检索词会两头不讨好：BM25 吃正文的话，表格里根本没有公司全称，
        # "迈瑞医疗2025年营业收入"会召回一堆别人家的表；吃 embed_text 的话，
        # 公司全称在每块里都出现，idf 被拉平，封面页反而排第一。
        # 实测：点名迈瑞的题，top-8 里 5 块来自别的公司。
        if only is None:
            ment = [c for c in self.companies if c in q]
            if ment:
                only = ment
        self.last_only = only

        if kinds:
            ks = set(kinds)
            allow = {i for i, c in enumerate(self.chunks) if c["kind"] in ks}
        else:
            allow = None
        if years:
            ys = set(years)
            allow = {i for i in (allow if allow is not None else range(len(self.chunks)))
                     if self.chunks[i]["year"] in ys}
        if only:
            os_ = set(only)
            allow = {i for i in (allow if allow is not None
                                 else range(len(self.chunks)))
                     if self.chunks[i]["name"] in os_}
            # per_company 是用来防止全景题被一家公司霸榜的。已经定向到具体公司了，
            # 再按公司去重就只剩 3 块——正好把那家公司的表挤掉。
            per_company = 10 ** 9
        # 多取一些再过滤：报告期一限，"年度"只占一半，公司一限可能只剩 1/12，
        # 取 50 条过滤完可能一条不剩
        CANDS = 2000 if allow is not None else 50

        def keep(lst):
            lst = [i for i in lst if allow is None or i in allow]
            return lst[:50]

        bl = keep([i for i, _ in self._bm25_cached(q, CANDS)])
        vl = keep([i for i, _ in self.search_vec(q, CANDS)]) if self.vectors is not None else []
        rb = {i: r + 1 for r, i in enumerate(bl)}
        rv = {i: r + 1 for r, i in enumerate(vl)}

        if base == "bm25":
            fused = [(i, 1.0 / (RRF_K + rb[i])) for i in bl]
        elif base == "vec":
            fused = [(i, 1.0 / (RRF_K + rv[i])) for i in vl]
        else:
            acc = {}
            for lst in (bl, vl):
                for r, i in enumerate(lst):
                    acc[i] = acc.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
            fused = sorted(acc.items(), key=lambda x: -x[1])

        if only:
            fused = [(i, s) for i, s in fused if self.chunks[i]["name"] in only]

        picked, cnt = [], {}
        for i, s in fused:
            n = self.chunks[i]["name"]
            if cnt.get(n, 0) >= per_company:
                continue
            cnt[n] = cnt.get(n, 0) + 1
            picked.append((i, s))
            if len(picked) >= topk:
                break

        out = []
        for i, s in picked:
            c = dict(self.chunks[i])
            c["_idx"] = i
            c["_score"] = round(s, 6)
            if trace:
                c["_rank_bm25"] = rb.get(i)
                c["_rank_vec"] = rv.get(i)
                c["_source"] = ("both" if i in rb and i in rv
                                else "bm25" if i in rb else "vec")
            out.append(c)
        return out


def mention_companies(q, companies):
    """问题里点名的公司（用于 only/定向）。"""
    return [c for c in companies if c in q]


if __name__ == "__main__":
    import time
    s = Searcher()
    print(f"{len(s.chunks)} 块，{len(s.companies)} 家公司，"
          f"向量 {'有' if s.vectors is not None else '无'}，"
          f"BM25 字段 {s.bm25_field} ({s.bm25_file})")
    q = sys.argv[1] if len(sys.argv) > 1 else "迈瑞医疗2025年的营业收入是多少"
    for mode in ("bm25", "vec", "mix"):
        if mode == "vec" and s.vectors is None:
            print("\n=== vec 跳过（索引未建）===")
            continue
        t0 = time.time()
        r = s.search(q, topk=5, mode=mode, trace=True)
        print(f"\n=== {mode} ({time.time()-t0:.2f}s) ===")
        for c in r:
            print(f"  {c['_score']:.5f}  {c['_source']:4}  bm25#{c['_rank_bm25'] or '-'}"
                  f"  vec#{c['_rank_vec'] or '-'}  {c['cite']}  {c['section'][:14]}"
                  f"  {c['text'][:50]!r}")
