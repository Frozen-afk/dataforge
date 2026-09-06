#!/usr/bin/env bash
#
# Start Latent Loop Lab and open it in a browser.
#
#   ./run.sh                 frontend + backend, opens the browser
#   ./run.sh --no-backend    frontend only
#   ./run.sh --no-open       do not launch a browser
#   ./run.sh --port 5180     serve the frontend somewhere else
#
# The artifact itself does not need the backend: both computational layers run
# in the browser from Frontend/public/data. The backend is started anyway
# because it is the reference implementation and its /docs page is a convenient
# way to poke at the mechanism. If it cannot start, this script says so and
# carries on rather than blocking the lesson.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$ROOT/Frontend"
BACKEND="$ROOT/Backend"

FRONTEND_PORT=5173
BACKEND_PORT=8000
START_BACKEND=1
OPEN_BROWSER=1
BACKEND_RUNNING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-backend) START_BACKEND=0; shift ;;
    --no-open)    OPEN_BROWSER=0; shift ;;
    --port)       FRONTEND_PORT="${2:?--port needs a number}"; shift 2 ;;
    -h|--help)    sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "Unknown option: $1. Try --help." >&2; exit 2 ;;
  esac
done

say()  { printf '\033[36m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# Each server is started inside a subshell, so $! is the subshell rather than
# the process holding the port: npm spawns vite, and killing npm's wrapper
# leaves vite listening. Every descendant has to go, or the next run hits
# "port already in use" on a server nobody can see.
PIDS=()

kill_tree() {
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_tree "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
  trap - INT TERM EXIT
  local pid
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill_tree "$pid"
  done

  # Give them a moment to close their sockets, then insist.
  sleep 0.5
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
  done

  echo
  say "Stopped."
}
trap cleanup INT TERM EXIT

port_busy() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3>&-
  fi
}

wait_for() {  # wait_for <port> <seconds>
  local port="$1" limit="$2" waited=0
  while ! port_busy "$port"; do
    sleep 0.3
    waited=$(( waited + 1 ))
    if (( waited > limit * 10 / 3 )); then
      return 1
    fi
  done
  return 0
}

# ---------------------------------------------------------------- frontend --

command -v node >/dev/null 2>&1 || die "node is not installed. The lab needs Node 18 or newer."

if [[ ! -d "$FRONTEND/node_modules" ]]; then
  say "Installing frontend dependencies (first run only)..."
  (cd "$FRONTEND" && npm install)
fi

if [[ ! -f "$FRONTEND/public/data/cases.json" ]]; then
  warn "No data bundle in Frontend/public/data."
  warn "Build it with:  cd Backend && python export_web.py"
fi

if port_busy "$FRONTEND_PORT"; then
  die "Port $FRONTEND_PORT is already in use. Try: ./run.sh --port 5180"
fi

say "Starting the lab on http://127.0.0.1:$FRONTEND_PORT"
# --host is not optional here. Left to itself Vite binds [::1] only, so the
# 127.0.0.1 URL this script prints and hands to the browser would be refused.
(cd "$FRONTEND" && npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort) &
PIDS+=($!)

# ----------------------------------------------------------------- backend --

if (( START_BACKEND )); then
  PYTHON=""
  for candidate in "$BACKEND/.venv/bin/python" "$(command -v python3 || true)"; do
    if [[ -x "$candidate" ]] && "$candidate" -c "import fastapi, uvicorn" 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  done

  if [[ -z "$PYTHON" ]]; then
    warn "Skipping the backend: no Python with FastAPI installed."
    warn "To enable it:  cd Backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  elif port_busy "$BACKEND_PORT"; then
    warn "Skipping the backend: port $BACKEND_PORT is already in use."
  else
    say "Starting the reference API on http://127.0.0.1:$BACKEND_PORT/docs"
    (cd "$BACKEND" && "$PYTHON" -m uvicorn server:app --host 127.0.0.1 --port "$BACKEND_PORT" --log-level warning) &
    PIDS+=($!)
    BACKEND_RUNNING=1
  fi
fi

# ------------------------------------------------------------------ browser --

if ! wait_for "$FRONTEND_PORT" 40; then
  die "The frontend did not come up on port $FRONTEND_PORT. Scroll up for the error."
fi

# Torch makes the backend slow to import, so it is still binding its port while
# the frontend is already serving. Wait before printing its URL rather than
# printing one that is not answering yet.
if (( BACKEND_RUNNING )) && ! wait_for "$BACKEND_PORT" 30; then
  warn "The reference API did not come up on port $BACKEND_PORT. The lab does not need it."
  BACKEND_RUNNING=0
fi

URL="http://127.0.0.1:$FRONTEND_PORT/#/start"

if (( OPEN_BROWSER )); then
  for opener in xdg-open open wslview; do
    if command -v "$opener" >/dev/null 2>&1; then
      "$opener" "$URL" >/dev/null 2>&1 &
      break
    fi
  done
fi

echo
say "Latent Loop Lab is running."
echo "  Lab           $URL"
if (( BACKEND_RUNNING )); then
  echo "  Reference API http://127.0.0.1:$BACKEND_PORT/docs"
fi
echo
echo "  Press Ctrl+C to stop."
echo

wait
