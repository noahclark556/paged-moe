#!/usr/bin/env bash
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
# Pre-bench cool-down: stop local inference, return Metal pages, report pressure.
#
# This is the fast path instead of a reboot. It cannot always zero swap
# (anonymous pages stick until their owners die or the machine reboots), but
# it clears the usual leftover from mlx / paged-moe sessions.
#
#   ./bench/cool_down.sh              # stop servers + Metal clear + report
#   ./bench/cool_down.sh --purge      # also sudo purge + reset wired_limit
#   ./bench/cool_down.sh --check      # report only (no stop / no clear)
#   ./bench/cool_down.sh --no-stop    # keep servers; Metal clear + report
#
# Exit 0 = ready enough for a bench. Exit 2 = still pressured; reboot.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$HOME/.mlx-env/bin/python" ]]; then
    PY="$HOME/.mlx-env/bin/python"
  else
    PY="$(command -v python3)"
  fi
fi

DO_STOP=1
DO_CLEAR=1
DO_PURGE=0
CHECK_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1; DO_STOP=0; DO_CLEAR=0; DO_PURGE=0 ;;
    --no-stop) DO_STOP=0 ;;
    --purge) DO_PURGE=1 ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown arg: $arg (try --help)" >&2
      exit 1
      ;;
  esac
done

# Ports a host app or a manual server normally owns.
PORTS=(8080 11434)

swap_used_mb() {
  # "vm.swapusage: total = 4096.00M  used = 3412.44M  free = ..."
  sysctl -n vm.swapusage 2>/dev/null \
    | awk '{ for (i=1;i<=NF;i++) if ($i=="used") { gsub(/M|,/,"",$(i+2)); print int($(i+2)+0.5); exit } }'
}

wired_limit_mb() {
  sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "?"
}

list_inference_pids() {
  # Match real server entrypoints only. Do not match this script's path
  # (.../paged-moe/bench/cool_down.sh) or the awk that lists them.
  # shellcheck disable=SC2009
  ps -ax -o pid=,command= 2>/dev/null | awk '
    /python.*pagedmoe\.server|python.*-m pagedmoe|mlx_lm\.server|mlx_openai|openai_server/ \
      && $0 !~ /(awk|cool_down\.sh)/ { print $1 }
  '
}

pids_on_port() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true
}

report() {
  local label="$1"
  local swap used free mem
  swap="$(sysctl -n vm.swapusage 2>/dev/null || echo "?")"
  used="$(swap_used_mb || echo 0)"
  echo ""
  echo "== $label =="
  echo "  swap:         $swap"
  echo "  wired_limit:  $(wired_limit_mb) MB"
  if command -v memory_pressure >/dev/null 2>&1; then
    mem="$(memory_pressure 2>/dev/null | awk -F': ' '/Pages free:/{print $2}' | head -1)"
    [[ -n "$mem" ]] && echo "  pages free:   $mem"
  fi
  local pids
  pids="$(list_inference_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  if [[ -n "$pids" ]]; then
    echo "  inference:    pids $pids"
  else
    echo "  inference:    none"
  fi
  for port in "${PORTS[@]}"; do
    local listeners
    listeners="$(pids_on_port "$port" | tr '\n' ',' | sed 's/,$//')"
    if [[ -n "$listeners" ]]; then
      echo "  :$port:         listening (pids $listeners)"
    fi
  done
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used >= 2048 )); then
    echo "  verdict:      PRESSURED (swap used ${used} MB)"
  elif [[ "$used" =~ ^[0-9]+$ ]] && (( used >= 512 )); then
    echo "  verdict:      soft pressure (swap used ${used} MB)"
  else
    echo "  verdict:      ok"
  fi
}

kill_pid() {
  local pid="$1"
  kill -TERM "$pid" 2>/dev/null || true
}

stop_servers() {
  echo ""
  echo "== stop =="
  if command -v ga >/dev/null 2>&1; then
    ga server stop 2>/dev/null || true
  fi

  local pid
  for pid in $(list_inference_pids); do
    echo "  TERM pid $pid (inference)"
    kill_pid "$pid"
  done
  for port in "${PORTS[@]}"; do
    for pid in $(pids_on_port "$port"); do
      echo "  TERM pid $pid (port $port)"
      kill_pid "$pid"
    done
  done

  sleep 1
  for pid in $(list_inference_pids); do
    echo "  KILL pid $pid"
    kill -KILL "$pid" 2>/dev/null || true
  done
  for port in "${PORTS[@]}"; do
    for pid in $(pids_on_port "$port"); do
      echo "  KILL pid $pid (port $port)"
      kill -KILL "$pid" 2>/dev/null || true
    done
  done
}

clear_metal() {
  echo ""
  echo "== metal =="
  # Short-lived process: import mlx, force wired_limit 0, dump the recycled pool.
  "$PY" - <<'PY'
import sys
try:
    import mlx.core as mx
except Exception as e:
    print(f"  skip: mlx not importable ({e})")
    sys.exit(0)

def gb(n):
    return round(n / 1e9, 3)

before_a = mx.get_active_memory()
before_c = mx.get_cache_memory()
if hasattr(mx, "set_wired_limit"):
    try:
        mx.set_wired_limit(0)
    except Exception as e:
        print(f"  set_wired_limit(0) failed: {e}")
if hasattr(mx, "set_cache_limit"):
    try:
        mx.set_cache_limit(0)
    except Exception:
        pass
mx.clear_cache()
after_a = mx.get_active_memory()
after_c = mx.get_cache_memory()
print(f"  mx active {gb(before_a)} -> {gb(after_a)} GB")
print(f"  mx cache  {gb(before_c)} -> {gb(after_c)} GB")
PY
}

do_purge() {
  echo ""
  echo "== purge (sudo) =="
  # Defends other mlx tools that still honor the raised wired limit.
  sudo sysctl iogpu.wired_limit_mb=0 || true
  # Drop file-backed page cache. Does not guarantee zero swap.
  if command -v purge >/dev/null 2>&1; then
    sudo purge
    echo "  purge done"
  else
    echo "  purge binary missing; skipped"
  fi
}

report "before"

if (( CHECK_ONLY )); then
  used="$(swap_used_mb || echo 0)"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used >= 2048 )); then
    echo ""
    echo "Still pressured. Reboot before trusting a bench."
    exit 2
  fi
  exit 0
fi

if (( DO_STOP )); then
  stop_servers
fi

if (( DO_CLEAR )); then
  clear_metal
fi

if (( DO_PURGE )); then
  do_purge
fi

# Give the kernel a beat to reclaim after process death.
sleep 1
report "after"

used="$(swap_used_mb || echo 0)"
echo ""
if [[ "$used" =~ ^[0-9]+$ ]] && (( used >= 2048 )); then
  echo "Still pressured (swap used ${used} MB)."
  echo "Cool-down got what it can; reboot before trusting tok/s numbers."
  exit 2
elif [[ "$used" =~ ^[0-9]+$ ]] && (( used >= 512 )); then
  echo "Soft pressure left (swap used ${used} MB)."
  echo "Fine for rough A/B; reboot if you need a clean absolute baseline."
  exit 0
else
  echo "Ready enough for a bench."
  exit 0
fi
