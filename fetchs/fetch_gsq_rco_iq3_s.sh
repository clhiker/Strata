#!/usr/bin/env bash
# Fetch ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (the IQ3_S mixed-precision build) into ~/models.
#
#   HF_ENDPOINT=https://hf-mirror.com ./fetch_gsq_rco_iq3_s.sh          # 默认：两个分片 + mmproj，并行、可续传
#   DEST=/path/to/dir ./fetch_gsq_rco_iq3_s.sh                          # 换目录
#   ./fetch_gsq_rco_iq3_s.sh --check                                    # 只校验已有文件，不下载
#
# 说明：huggingface.co 在本机不可达，且 xet 的 CAS 桥会返回 403，所以这里走纯 HTTP 的
# resolve/main + curl -C -（与 fetch_orca_iq3_xxs.sh 同样的路子），逐文件校验精确字节数。
set -uo pipefail

REPO="${REPO:-ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF}"
REVISION="${REVISION:-main}"
DEST="${DEST:-/home/aipl/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S}"
EP="${HF_ENDPOINT:-https://hf-mirror.com}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# "<仓库内路径>|<本地文件名>|<字节数>"
FILES=(
  "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf|Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf|54817524224"
  "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf|Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf|28800138432"
  "mmproj-Qwen3.8-Flash-Next-BF16.gguf|mmproj-Qwen3.8-Flash-Next-BF16.gguf|907543008"
  "tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt|Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt|0"
)

mkdir -p "$DEST" || exit 1
echo "目标目录: $DEST"
echo "镜像    : $EP"

size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }

fetch_one() {
  local remote="$1" local_name="$2" want="$3" out="$DEST/$2" url part got
  url="$EP/$REPO/resolve/$REVISION/$remote"
  if [ "$want" != "0" ] && [ "$(size_of "$out")" = "$want" ]; then
    echo "  ✓ $local_name 已完整（$(numfmt --to=iec "$want")）"
    return 0
  fi
  [ "$CHECK_ONLY" = 1 ] && { echo "  ✗ $local_name 缺失或大小不对（期望 ${want:-?}）"; return 1; }
  part="$out.part"
  echo "  ↓ $local_name"
  # -C - 断点续传；--retry-all-errors 让 403/超时也重试（xet 403 就是靠这个扛过去的场景不会出现，走的是 HTTP 路径）
  curl -sSL --retry 40 --retry-delay 5 --retry-all-errors -C - -o "$part" "$url" || { echo "  ✗ $local_name 下载中断（可重跑续传）"; return 1; }
  got="$(size_of "$part")"
  if [ "$want" != "0" ] && [ "$got" != "$want" ]; then
    echo "  ✗ $local_name 大小不符：得到 $got，期望 $want"
    return 1
  fi
  mv -f "$part" "$out"
  echo "  ✓ $local_name 完成（$(numfmt --to=iec "$got")）"
}

rc=0
for spec in "${FILES[@]}"; do
  IFS='|' read -r remote local_name want <<<"$spec"
  # 大件（>1 GiB）并行后台跑，小件前台
  if [ "$want" != "0" ] && [ "$want" -gt 1073741824 ] && [ "$CHECK_ONLY" = 0 ]; then
    fetch_one "$remote" "$local_name" "$want" &
  else
    fetch_one "$remote" "$local_name" "$want" || rc=1
  fi
done
wait || rc=1

echo
echo "=== 校验 ==="
for spec in "${FILES[@]}"; do
  IFS='|' read -r remote local_name want <<<"$spec"
  got="$(size_of "$DEST/$local_name")"
  if [ "$want" = "0" ] || [ "$got" = "$want" ]; then
    printf '  ✓ %-58s %s\n' "$local_name" "$(numfmt --to=iec "$got")"
  else
    printf '  ✗ %-58s %s (期望 %s)\n' "$local_name" "$got" "$want"; rc=1
  fi
done
exit $rc
