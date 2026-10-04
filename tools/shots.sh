#!/usr/bin/env bash
# 一次性截完交付要用的页面图。跑之前先起页面：
#   python src/app.py --port 8760
# 注意 06/07 是同一道题在两套检索下的对照，要一起看才有意义。
set -u
cd "$(dirname "$0")/.."
P=${PORT:-8760}

# 有回答的图统一裁到 4600 逻辑像素。召回的正文块很长，整页能有 7000+，
# 缩进文档里字就糊了；4600 大致是"筛选栏 + 回答 + 前几块材料"。
CAP=4600

shot () {  # shot <名字> <检索方式> <问题> [topk]
  local name="$1" mode="$2" q="$3" topk="${4:-}"
  echo "===== $name ($mode${topk:+ · 召回 $topk}) ====="
  # 分成两支是因为 ${4:+--topk "$4"} 那种写法里引号会被当字面量，传参不可靠
  if [ -n "$topk" ]; then
    python tools/shot.py --port "$P" --shot "$name" --mode "$mode" \
      --topk "$topk" --max-px "$CAP" --q "$q" || echo "  !! $name 失败"
  else
    python tools/shot.py --port "$P" --shot "$name" --mode "$mode" \
      --max-px "$CAP" --q "$q" || echo "  !! $name 失败"
  fi
}

# 首屏（不提问）
python tools/shot.py --port "$P" --shot 00-首屏

shot 01-单公司三数·混合 mix  "迈瑞医疗 2025 年的营业收入、归母净利润、扣非归母净利润分别是多少？"
shot 02-单公司增速·混合 mix  "联影医疗 2025 年的营业收入和归母净利润同比增长率分别是多少？"
shot 03-语义检索·混合   mix  "哪些公司的 2025 年报里提到了集采（带量采购）？它们分别是怎么描述的？"
shot 04-口径陷阱·混合   mix  "大博医疗 2025 年的净利润是多少？"

# 同一道题，两套检索的对照：增速题是 BM25 的词汇鸿沟、向量该赢的那一类
shot 05-增速题·仅BM25  bm25 "联影医疗 2025 年的营业收入和归母净利润同比增长率分别是多少？"
shot 06-增速题·仅向量  vec  "联影医疗 2025 年的营业收入和归母净利润同比增长率分别是多少？"

# 全景题前后对照：同一道题，普通混合检索只能盖住几家，
# 扇出后 12 家全到——这两张要挨着看
shot 07-全景题·混合    mix       "2025 年度归母净利润同比下滑的有哪几家公司？" 8
shot 08-全景题·扇出    panorama  "2025 年度归母净利润同比下滑的有哪几家公司？" 24

echo
echo "全部完成，图在 docs/shots/"
