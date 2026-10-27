#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$HOME/AlbertoDir/DT_sensing_fusion"
cd "$PROJECT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
  echo "[ERROR] No existe el entorno virtual: $PROJECT_DIR/.venv"
  exit 1
fi

source .venv/bin/activate

mkdir -p logs

TS="$(date +%Y%m%d_%H%M%S)"
PY_LOG="logs/online_5g_python_${TS}.log"
MATLAB_LOG="logs/online_5g_matlab_${TS}.log"

PY_PID=""

cleanup() {
  echo ""
  echo "[STOP] Cerrando demo online 5G..."

  if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
    echo "[STOP] Matando servidor Python PID=${PY_PID}"
    kill "${PY_PID}" 2>/dev/null || true
    sleep 1
    kill -9 "${PY_PID}" 2>/dev/null || true
  fi

  echo "[STOP] Hecho."
}

trap cleanup EXIT INT TERM

echo "============================================================"
echo "ONLINE 5G BINARY EMPTY/P5 + JSON + SCP"
echo "============================================================"
echo "Proyecto: $PROJECT_DIR"
echo "Python log: $PY_LOG"
echo "MATLAB log: $MATLAB_LOG"
echo ""
echo "JSON local:"
echo "  results/online/live_inference_state_5G.json"
echo ""
echo "JSON remoto por SCP:"
echo "  nextnet@163.117.140.146:~/AlbertoDir/demo_5G/5G_inference/live_inference_state_5G.json"
echo "============================================================"
echo ""

echo "[1/2] Lanzando servidor Python de inferencia..."
python -u src/python/online_rxgridssb_inference_server.py > "$PY_LOG" 2>&1 &
PY_PID=$!

echo "[INFO] Python PID=${PY_PID}"
echo "[INFO] Esperando ${PYTHON_STARTUP_SLEEP:-8} segundos para que el servidor arranque..."
sleep "${PYTHON_STARTUP_SLEEP:-8}"

if ! kill -0 "$PY_PID" 2>/dev/null; then
  echo "[ERROR] Python se ha cerrado antes de lanzar MATLAB."
  echo "Últimas líneas del log:"
  tail -n 120 "$PY_LOG"
  exit 1
fi

echo ""
echo "[2/2] Lanzando MATLAB streamer..."
echo "[INFO] SEND_EVERY_N=${SEND_EVERY_N:-1}"

if [[ -n "${MAX_VALID_SSB:-}" ]]; then
  echo "[INFO] MAX_VALID_SSB=${MAX_VALID_SSB}"
else
  echo "[INFO] MAX_VALID_SSB=Inf"
fi

echo ""

export SEND_EVERY_N="${SEND_EVERY_N:-1}"

if [[ -n "${MAX_VALID_SSB:-}" ]]; then
  export MAX_VALID_SSB
fi

matlab -batch "run('src/matlab/stream_rxgridssb_online_to_python.m')" 2>&1 | tee "$MATLAB_LOG"

echo ""
echo "[DONE] MATLAB ha terminado."
