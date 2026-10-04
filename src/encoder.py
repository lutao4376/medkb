# -*- coding: utf-8 -*-
"""
自己实现的批量编码，替掉 sentence-transformers 的 encode。

为什么不用 ST：
  实测 ST 每次 encode 调用都有一笔很大的固定开销，切 504 次调用光开销就一两个小时。
  而且它对 padding 的处理不如这里可控。
本实现：
  * 按长度排序后组批——同一批里长度接近，padding 几乎为零；
  * 按 token 预算组批而不是按条数，显存占用平稳，不会在大块上突然爆掉；
  * Qwen3-Embedding 用 last-token pooling（左 padding），配 L2 归一化。

正确性有对拍：`python src/encoder.py --verify` 拿 ST 的结果比余弦相似度。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\hf-cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"

MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAXLEN = 1536
TOKEN_BUDGET = 45000     # 每批的总 token 上限（6.4G 显存实测到 6 万就顶到 95%，留出余量）


class Encoder:
    def __init__(self, model=MODEL, device=None, half=True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tk = AutoTokenizer.from_pretrained(model, padding_side="left")
        dtype = torch.float16 if (half and device == "cuda") else torch.float32
        self.m = AutoModel.from_pretrained(model, dtype=dtype).to(device).eval()
        self.dim = self.m.config.hidden_size

    def _forward(self, texts, idx, maxlen):
        torch = self.torch
        enc = self.tk([texts[i] for i in idx], padding=True,
                      truncation=True, max_length=maxlen, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            h = self.m(**enc).last_hidden_state
        # 左 padding -> 最后一位就是真实末 token
        v = torch.nn.functional.normalize(h[:, -1].float(), dim=-1)
        return v.cpu().numpy(), int(enc["input_ids"].numel())

    def encode(self, texts, maxlen=MAXLEN, budget=TOKEN_BUDGET, log=None, on_batch=None):
        """
        on_batch(done_items, n_items, done_tok, total_tok, elapsed_s) —— 每批回调一次，
        调用方用它写进度文件给进度条看。
        """
        torch = self.torch
        n = len(texts)
        out = np.zeros((n, self.dim), dtype=np.float32)

        # 先只取长度（不编码），按长度排序后组批 —— padding 最小
        lens = [len(self.tk(t, truncation=True, max_length=maxlen)["input_ids"]) for t in texts]
        order = sorted(range(n), key=lambda i: lens[i])

        batches, cur, curmax = [], [], 0
        for i in order:
            # 加进来之后这一批要补齐到 max(curmax, len_i)
            newmax = max(curmax, lens[i])
            if cur and newmax * (len(cur) + 1) > budget:
                batches.append(cur)
                cur, curmax = [i], lens[i]
            else:
                cur.append(i)
                curmax = newmax
        if cur:
            batches.append(cur)

        pad_tok = 0
        real_tok = sum(lens)
        splits = 0
        pending = list(batches)
        done_items, done_tok = 0, 0
        t0 = time.time()
        while pending:
            idx = pending.pop(0)
            try:
                v, pt = self._forward(texts, idx, maxlen)
            except RuntimeError as ex:
                # 6.4G 显存很紧，一批撑爆就劈成两半重来，别让整个任务崩掉
                if "out of memory" not in str(ex).lower() or len(idx) <= 1:
                    raise
                torch.cuda.empty_cache()
                h = len(idx) // 2
                pending.insert(0, idx[h:])
                pending.insert(0, idx[:h])
                splits += 1
                continue
            out[idx] = v
            pad_tok += pt
            done_items += len(idx)
            done_tok += sum(lens[i] for i in idx)
            if on_batch:
                on_batch(done_items, n, done_tok, real_tok, time.time() - t0)
            if log and (done_items == n or done_items // 512 != (done_items - len(idx)) // 512):
                el = time.time() - t0
                rate = done_tok / el if el else 0
                eta = (real_tok - done_tok) / rate if rate else 0
                print(f"    {done_items}/{n}  {rate:.0f} tok/s  "
                      f"已用{el/60:.1f}分  预计还剩{eta/60:.1f}分"
                      + (f"  (拆分 {splits} 次)" if splits else ""),
                      file=log, flush=True)
        self.stats = {"real_tok": real_tok, "padded_tok": pad_tok,
                      "waste": pad_tok / max(1, real_tok) - 1, "batches": len(batches),
                      "splits": splits}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="和 sentence-transformers 对拍")
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not args.verify:
        return
    cs = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
    texts = [c["embed_text"] for c in cs[: args.sample]]

    e = Encoder()
    t0 = time.time()
    mine = e.encode(texts, log=sys.stdout)
    el = time.time() - t0
    print(f"\n自研编码 {args.sample} 块: {el:.1f}s  {e.stats['real_tok']/el:.0f} tok/s")
    print(f"  padding 浪费 {e.stats['waste']*100:.1f}%  批次 {e.stats['batches']}"
          f"  真实 token {e.stats['real_tok']}")

    from sentence_transformers import SentenceTransformer
    sm = SentenceTransformer(MODEL, device=e.device)
    sm.max_seq_length = MAXLEN
    sm.half()
    t0 = time.time()
    ref = sm.encode(texts, batch_size=64, normalize_embeddings=True,
                    show_progress_bar=False, convert_to_numpy=True)
    st_el = time.time() - t0
    print(f"ST 编码 {args.sample} 块: {st_el:.1f}s")

    sim = (mine * ref).sum(axis=1)
    print(f"\n对拍余弦相似度: 最小 {sim.min():.4f}  中位 {np.median(sim):.4f}  "
          f"<0.99 的 {(sim<0.99).sum()} 个")
    print(f"速度提升 {st_el/el:.1f}×")


if __name__ == "__main__":
    main()
