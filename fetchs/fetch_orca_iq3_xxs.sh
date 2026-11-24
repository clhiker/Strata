#!/usr/bin/env bash
# Fetch OrcaRouter Qwen3.8-Flash-Next-Uncensored IQ3_XXS into ~/models, for Strata (docs/ORCA.md).
#
#   ./fetch_orca_iq3_xxs.sh --check      # preflight only: gating, disk space, sizes (no download)
#   ./fetch_orca_iq3_xxs.sh --dry-run    # print the exact download commands
#   ./fetch_orca_iq3_xxs.sh              # download the two IQ3_XXS shards
#   ./fetch_orca_iq3_xxs.sh --with-mtp   # + the MTP draft head (4.14 GB)
#
# The repo is GATED (gated: auto): accept the terms on its Hugging Face page once, then
#   export HF_TOKEN=hf_xxx
# huggingface.co is unreachable from this host (connection timeout), so hf-mirror.com is the default host.
set -euo pipefail

REPO="${REPO:-orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF}"
REVISION="${REVISION:-main}"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DEST="${DEST:-$HOME/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS}"

# The q3 build Strata validates for this repo (docs/ORCA.md). IQ3_M / Q3_K_* are NOT usable:
# IQ3_M's expert down matrices are Q5_0, which the native GPU expert path does not support.
SHARD1="Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf"
SHARD2="Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf"
MTP="Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf"
SIZE_SHARD1=44637691008
SIZE_SHARD2=40564977024
SIZE_MTP=4135893632

WITH_MTP=0; MODE=run

usage() {
  cat <<EOF
用法: $(basename "$0") [-h|--help] [--check|--dry-run] [--with-mtp]

把 OrcaRouter Qwen3.8-Flash-Next-Uncensored 的 IQ3_XXS 拉进 $DEST。
（Strata 唯一验证过的 q3 档：IQ3_M 的专家 down 是 Q5_0、Q3_K_* 是 K-quant，引擎都用不了。）

选项:
  -h, --help     显示本帮助
  --check        只做预检（门禁 / 磁盘 / 体积 / 令牌），不下载任何东西
  --dry-run      打印将要执行的 hf download 命令，不下载
  --with-mtp     额外拉 MTP 草稿头（Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf, 3.85 GiB）
  (无参数)       下载两个 IQ3_XXS 分片

下载内容与体积:
  00001-of-00002.gguf   41.57 GiB   （分片 1：含 PLE 表，引擎的 --native/--ple-gguf 都指向它）
  00002-of-00002.gguf   37.78 GiB
  MTP-draft.gguf         3.85 GiB   仅 --with-mtp
  合计                  79.35 GiB（+3.85 with --with-mtp）；下载完逐个校验字节数

环境变量:
  HF_TOKEN=hf_xxx   必需。该仓库是 gated（gated: auto）：先在仓库页点接受条款，再给令牌
  HF_ENDPOINT       HF 主机，默认 https://hf-mirror.com
                    （本机直连 huggingface.co 会 443 超时；镜像同样受门禁约束，权限在 HF 侧判定）
  DEST              落地目录，默认 $HOME/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS
  REPO              仓库 id，默认 $REPO
  REVISION          分支/提交，默认 main

行为: 走 \`hf download\`，断点续传（.incomplete 可续）；开始前查门禁与磁盘（payload + 5% 余量）；
      结束后逐个校验文件字节数。任何一步失败都会明确说清是哪一步。

下一步（拉完之后，见 docs/ORCA.md）:
  .venv/bin/python tools/iq_pack.py --gguf <分片1> --out packs/orca-iq3_xxs --compat-bf16
EOF
}

for a in "$@"; do
  case "$a" in
    --with-mtp) WITH_MTP=1 ;;
    --check|--dry-run) MODE="${a#--}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $a (try -h)" >&2; exit 2 ;;
  esac
done

INCLUDE=("$SHARD1" "$SHARD2"); WANT=$((SIZE_SHARD1 + SIZE_SHARD2))
[ "$WITH_MTP" = 1 ] && { INCLUDE+=("$MTP"); WANT=$((WANT + SIZE_MTP)); }

human() { awk -v b="$1" 'BEGIN{printf "%.2f GiB", b/1073741824}'; }
say() { printf '  %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

printf 'OrcaRouter IQ3_XXS -> %s\n' "$DEST"
say "host    : $ENDPOINT  (HF_ENDPOINT)"
say "revision: $REVISION"
say "payload : $(printf '%s ' "${INCLUDE[@]}")= $(human $WANT)"

# ---- preflight -------------------------------------------------------------
command -v hf >/dev/null || die "the 'hf' CLI is missing (pip install -U huggingface_hub)"
TOKEN_ARGS=()
[ -n "${HF_TOKEN:-}" ] && TOKEN_ARGS=(--token "$HF_TOKEN")
say "token   : ${HF_TOKEN:+set}${HF_TOKEN:-not set}"

probe=$(curl -sS -o /dev/null -w '%{http_code}' -m 25 \
        ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
        "$ENDPOINT/$REPO/resolve/$REVISION/.gitattributes" || echo 000)
case "$probe" in
  200) say "gating  : OK (HTTP 200, a token is accepted or the probe passed)" ;;
  401|403) die "the repo is gated and this request was refused (HTTP $probe).
       Accept the terms at https://huggingface.co/$REPO , then 'export HF_TOKEN=hf_...' (read token) and retry." ;;
  000) die "no connection to $ENDPOINT (set HF_ENDPOINT to a host that works here)" ;;
  *)   die "unexpected HTTP $probe from $ENDPOINT" ;;
esac

mkdir -p "$DEST"
avail=$(df -Pk "$DEST" | awk 'NR==2{print $4}')
need_kb=$(( WANT / 1024 + WANT / 1024 / 20 ))          # payload + 5% headroom
say "disk    : $(human $((avail*1024))) free at $DEST, need $(human $((need_kb*1024)))"
[ "$avail" -ge "$need_kb" ] || die "not enough free space for this download"
[ "$MODE" = check ] && { say "preflight only (--check): nothing downloaded."; exit 0; }

# ---- download (resumable; hf keeps .incomplete parts) ----------------------
CMD=(hf download "$REPO" --revision "$REVISION" --local-dir "$DEST"
     --include "${INCLUDE[@]}" "${TOKEN_ARGS[@]}")
export HF_ENDPOINT="$ENDPOINT"
if [ "$MODE" = dry-run ]; then
  printf '\nHF_ENDPOINT=%s \\\n  ' "$ENDPOINT"; printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi
printf '\n'; "${CMD[@]}"

# ---- verify ----------------------------------------------------------------
fail=0
check() { # name expected
  local f="$DEST/$1" got
  [ -f "$f" ] || { say "MISSING $1"; fail=1; return; }
  got=$(stat -c %s "$f")
  if [ "$got" = "$2" ]; then say "OK   $1 ($(human "$got"))"
  else say "SIZE $1 is $got, expected $2 (truncated or still downloading)"; fail=1; fi
}
printf '\n'; check "$SHARD1" "$SIZE_SHARD1"; check "$SHARD2" "$SIZE_SHARD2"
[ "$WITH_MTP" = 1 ] && check "$MTP" "$SIZE_MTP"
[ "$fail" = 0 ] || die "verification failed; rerun this script (it resumes)"

cat <<EOF

Done. Next, from the Strata checkout (docs/ORCA.md) - a .venv does not exist yet:
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python tools/iq_pack.py \\
      --gguf "$DEST/$SHARD1" --out packs/orca-iq3_xxs --compat-bf16
  # then the config + server in docs/ORCA.md (----native/--ple-gguf = the same shard 1, 32K context, --kv int8)
EOF
