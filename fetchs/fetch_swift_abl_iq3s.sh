#!/usr/bin/env bash
# 下载 SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF 的 IQ3_S 档
#   = ukisai Swift-1.5 的 GSQ 量化 + orcarouter Uncensored 的 144 张量移植（去审查），自带 Strata 草稿运行时
# 走 hf-mirror（huggingface.co 直连不可达；代理 8118 只有 2.8 MB/s，镜像 4 路 12.8 MB/s）
#   ./fetch_swift_abl_iq3s.sh            # 下载（可反复重跑，断点续传）
#   ./fetch_swift_abl_iq3s.sh --check    # 只校验
set -uo pipefail
REPO="${REPO:-SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF}"
REV="${REV:-main}"
DEST="${DEST:-/home/aipl/models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S}"
EP="${HF_ENDPOINT:-https://hf-mirror.com}"
ARIA="${ARIA:-$HOME/.local/bin/aria2c}"
mkdir -p "$DEST"
FILES=(
"Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf|44849330016"
"Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf|38699929024"
"mmproj-Qwen3.8-Flash-Next-BF16.gguf|907542944"
"mtp-q2_0.gguf|889017216"
"strata/rt/dense.bin|116099072"
"strata/rt/dense.txt|1880"
"strata/rt/draft_vocab.bin|425196"
"strata/rt/experts.bin|707788800"
)
fetch() { # $1=相对路径 $2=字节
  local n="$1" want="$2" p="$DEST/$1"
  mkdir -p "$(dirname "$p")"
  [ -f "$p" ] && [ "$(stat -c %s "$p")" = "$want" ] && { echo "  ✓ 已有 $n"; return 0; }
  echo "  ↓ $n"
  "$ARIA" -x4 -s4 -k1M -c --file-allocation=none --all-proxy="" --no-conf --console-log-level=warn \
          --summary-interval=0 --max-tries=40 --retry-wait=5 \
          -d "$(dirname "$p")" -o "$(basename "$p")" "$EP/$REPO/resolve/$REV/$n" >/dev/null 2>&1
  [ "$(stat -c %s "$p" 2>/dev/null)" = "$want" ] && echo "  ✓ $n" || { echo "  ✗ $n 大小不符"; return 1; }
}
if [ "${1:-}" = "--check" ]; then
  ok=0; bad=0
  for e in "${FILES[@]}"; do n="${e%%|*}"; w="${e##*|}"; s=$(stat -c %s "$DEST/$n" 2>/dev/null || echo 0)
    if [ "$s" = "$w" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "  ✗ $n 得到 $s 期望 $w"; fi; done
  echo "  校验: $ok 个完整, $bad 个缺失/不符"; [ "$bad" = 0 ] || exit 1; exit 0
fi
echo "目标: $DEST（镜像 $EP）"
fetch "strata/rt/dense.bin" 116099072 & fetch "strata/rt/experts.bin" 707788800 & fetch "strata/rt/draft_vocab.bin" 425196 & fetch "strata/rt/dense.txt" 1880 &
fetch "mmproj-Qwen3.8-Flash-Next-BF16.gguf" 907542944 & fetch "mtp-q2_0.gguf" 889017216 &
wait
fetch "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf" 44849330016 &
fetch "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf" 38699929024 &
wait
echo "=== 校验 ==="; "$0" --check
