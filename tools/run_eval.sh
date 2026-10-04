#!/usr/bin/env bash
# 向量建好之后，把该跑的评测一次跑完（不截图，截图要另外起页面）。
#
#   bash tools/run_eval.sh 2>&1 | tee logs/eval_all.log
#
# 三步都是幂等的：结果写进 out/，重跑覆盖。qa.py 每题调一次模型、约 35 秒，
# 10 题的 run 大约 6 分钟，三个 run 十几分钟——别在掐表等的时候跑。
set -u
cd "$(dirname "$0")/.."

if [ ! -f data/index/vectors.npy ]; then
  echo "还没有 data/index/vectors.npy，先跑 python src/index.py"
  exit 1
fi

echo "=== 0/4 向量已就位：$(stat -c%s data/index/vectors.npy) 字节 ==="

echo
echo "=== 1/4 消融表（不调模型）==="
python src/bench_retrieval_ablation.py --with-vec

echo
echo "=== 2/4 纯检索评测：三路 × 报告期自动 ==="
python src/eval_retrieval.py --modes bm25 vec mix panorama --kinds auto --topk 8

echo
echo "=== 3/4 调模型跑 10 题 ==="
# 基线：作业默认口径，单轮 top-8。失败就记在这里，不改。
python src/qa.py --questions data/questions.json --mode mix --topk 8 \
  --out out/qa_mix.json
# BM25 单路：用来看"哪一路找到的"，以及词汇鸿沟到底伤到哪几题
python src/qa.py --questions data/questions.json --mode bm25 --topk 8 \
  --out out/qa_bm25.json
# 改进：全景扇出 + 每家公司取 2 块（24/12）
python src/qa.py --questions data/questions.json --mode panorama --topk 24 \
  --out out/qa_pano.json

echo
echo "=== 4/4 判分与逐题溯源 ==="
python src/score.py --runs out/qa_bm25.json out/qa_mix.json out/qa_pano.json \
  --md out/检索评测.md

echo
echo "全部完成。截图另跑：bash tools/shots.sh"
