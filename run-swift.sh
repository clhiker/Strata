#!/usr/bin/env bash
# 一键启动 Strata + Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated (IQ3_S)
#   = ukisai Swift-1.5 的 GSQ-RCO 量化 + orcarouter Uncensored 的 144 张量移植（去审查）
#   自带 Strata 草稿运行时（--mtp 指向模型目录里的 strata/rt）
#
# 引擎: 0.1.60.1（v0.2.14.1，构建目录 build/，二进制 build/strata）
#       回退二进制: engine-bak/strata-0.1.56、engine-bak/strata-0.1.60
#       0.1.60 的 Turing 融合专家内核命中本 pack 的 IQ3_XXS/IQ2_S + IQ4_NL 对：
#       128K 预填 +3.7~4.8%、341K +5.9%，decode 持平（详见 docs/LOCAL-FINDINGS.zh.md）
#
# 模型名: qwen-flash    监听: 0.0.0.0:1234    baseURL: http://127.0.0.1:1234/v1
#   ./run-swift.sh                    # 256K 原生（默认）
#   ./run-swift.sh --context 512k     # 512K（yarn/2）
#   ./run-swift.sh --context 1m       # 1M（yarn/4，KV streaming）
#   ./run-swift.sh --status|--stop|--restart
#   ./run-swift.sh --check-paths
#   CONFIG=configs/strata-swift-abl-iq3s-512k.json ./run-swift.sh
# 其余参数原样传给 serve.server（如 --port 1234）。
set -euo pipefail
cd "$(dirname "$0")"

BASE_256="configs/strata-swift-abl-iq3s-256k.json"
BASE_512="configs/strata-swift-abl-iq3s-512k.json"
BASE_1M="configs/strata-swift-abl-iq3s-1m.json"
MODE=start
CTX=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
      exit 0 ;;
    --context)
      [[ $# -ge 2 ]] || { echo "ERROR: --context needs 256k, 512k or 1m" >&2; exit 2; }
      CTX="$2"; shift 2 ;;
    --status) MODE=status; shift ;;
    --stop) MODE=stop; shift ;;
    --restart) MODE=restart; shift ;;
    --check-paths) MODE=check; shift ;;
    --) shift; ARGS+=("$@"); break ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -n "${CONFIG:-}" ]]; then
  CFG="$CONFIG"
else
  case "$CTX" in
    ""|256k|262144) CFG="$BASE_256" ;;
    512k|524288) CFG="$BASE_512" ;;
    1m|1048576) CFG="$BASE_1M" ;;
    *) echo "ERROR: --context $CTX: use 256k, 512k or 1m" >&2; exit 2 ;;
  esac
fi
[[ -f "$CFG" ]] || { echo "ERROR: config not found: $CFG" >&2; exit 1; }
CFG_ABS="$(readlink -f "$CFG")"

running_pids() {
  ps -eo pid=,args= | awk -v cfg="$CFG" -v abs="$CFG_ABS" '
    {
      pid=$1; line=$0; sub(/^[ \t]*[0-9]+[ \t]+/, "", line)
      n=split(line, t, /[ \t]+/)
      if (t[1] !~ /(^|\/)python[0-9.]*$/ || index(line, "-m serve.server") == 0) next
      c=""
      for (i=1; i<n; i++) if (t[i] == "--config") c=t[i+1]
      if (c == cfg || c == abs) print pid, c
    }'
}

running_any() {
  ps -eo pid=,args= | awk '
    {
      pid=$1; line=$0; sub(/^[ \t]*[0-9]+[ \t]+/, "", line)
      n=split(line, t, /[ \t]+/)
      if (t[1] !~ /(^|\/)python[0-9.]*$/ || index(line, "-m serve.server") == 0) next
      c=""
      for (i=1; i<n; i++) if (t[i] == "--config") c=t[i+1]
      if (c ~ /strata-swift-abl-iq3s/) print pid, c
    }'
}

select_pids() {
  if [[ -n "${CONFIG:-}" || -n "$CTX" ]]; then running_pids; else running_any; fi
}

config_value() {
  .venv/bin/python - "$1" "$2" <<'PY' 2>/dev/null || true
import json, sys
print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))
PY
}

config_port() { config_value "$1" port; }

listening_port() {
  ss -ltnp 2>/dev/null | awk -v p="pid=$1," '
    index($0, p) { n=split($4, a, ":"); print a[n]; exit }'
}

status() {
  local rows; rows="$(select_pids)"
  local exe; exe="$(config_value "$CFG" exe)"
  if [[ -n "$exe" ]]; then
    if [[ -x "$exe" ]]; then
      echo "engine: $("$exe" --version 2>/dev/null | head -1)  ($exe)"
    else
      echo "engine: MISSING exe  ($exe)"
    fi
  fi
  if [[ -z "$rows" ]]; then echo "not running"; return 0; fi
  echo "$rows" | while read -r pid cfg; do
    [[ -z "${pid:-}" ]] && continue
    local port; port="$(listening_port "$pid")"
    [[ -z "$port" ]] && port="$(config_port "$cfg")"
    echo "running: pid $pid  config $cfg"
    if [[ -n "$port" ]] && curl -sf -m 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "  ready: http://127.0.0.1:$port/v1"
    else
      echo "  not responding on port ${port:-?} (check the config's log)"
    fi
  done
}

kill_tree() {
  local pid="$1" kid
  for kid in $(pgrep -P "$pid" 2>/dev/null || true); do kill_tree "$kid"; done
  kill "$pid" 2>/dev/null || true
}

stop() {
  # 同一时间只跑一个 Swift 服务；切档时也要停掉当前档，而不只是目标档
  local pids; pids="$(running_any | awk '{print $1}')"
  if [[ -z "$pids" ]]; then echo "not running"; return 0; fi
  for pid in $pids; do echo "stopping pid $pid"; kill_tree "$pid"; done
  for _ in $(seq 1 20); do
    local alive=0
    for pid in $pids; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [[ "$alive" == 0 ]] && { echo "stopped"; return 0; }
    sleep 1
  done
  echo "still running; sending SIGKILL" >&2
  for pid in $pids; do kill -9 "$pid" 2>/dev/null || true; done
}

check_paths() {
  .venv/bin/python - "$CFG" <<'PY'
import json, os, subprocess, sys
c = json.load(open(sys.argv[1])); a = c.get("args", [])
def g(k):
    return a[a.index(k) + 1] if k in a else None
items = [("engine.exe", c.get("exe")), ("native", g("--native")), ("pack", g("--pack")),
         ("mtp", g("--mtp")), ("ple-gguf", g("--ple-gguf"))]
v = c.get("vision") or {}
items += [("vision.exe", v.get("exe")), ("vision.mmproj", v.get("mmproj")),
          ("vision.model", v.get("model"))]
bad = 0
for name, path in items:
    if not path:
        continue
    ok = os.path.exists(path)
    print(("OK   " if ok else "MISS ") + name + ": " + path)
    bad += 0 if ok else 1
exe = c.get("exe")
if exe and os.access(exe, os.X_OK):
    try:
        ver = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
        print("engine version: " + (ver or "unknown"))
    except Exception as e:
        print("engine version: unknown (" + str(e) + ")")
sys.exit(1 if bad else 0)
PY
}

case "$MODE" in
  status) status; exit 0 ;;
  stop) stop; exit 0 ;;
  check) check_paths; exit $? ;;
  restart) stop; sleep 1 ;;
esac

check_paths >/dev/null 2>&1 || { echo "ERROR: a model/pack path is missing in $CFG (run --check-paths)" >&2; exit 1; }
CFG_PORT="$(config_value "$CFG" port)"
CFG_HOST="$(config_value "$CFG" host)"
JOINED=" ${ARGS[*]:-} "
if [[ -n "$CFG_PORT" && "$JOINED" != *" --port "* ]]; then ARGS+=(--port "$CFG_PORT"); fi
if [[ -n "$CFG_HOST" && "$JOINED" != *" --host "* ]]; then ARGS+=(--host "$CFG_HOST"); fi
echo "starting Strata: $CFG"
# v0.2.16 布局：server 在 terminal/serve（包根 terminal/），配置里的相对路径仍按仓库根解析
export PYTHONPATH="$PWD/terminal${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m serve.server --engine strata --config "$CFG" ${ARGS[@]+"${ARGS[@]}"}
