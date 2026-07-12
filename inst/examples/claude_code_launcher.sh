#!/usr/bin/env bash
# Launch a local llamaR Anthropic server and start Claude Code against it.
#
#   bash inst/examples/claude_code_launcher.sh [model.gguf] [port] [-- claude args...]
#
# Starts llama_serve_anthropic() in the background, waits until it answers,
# then runs `claude` with ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY pointed at it.
# When claude exits (or you Ctrl-C), the server is stopped automatically.
#
# By default this loads BOTH models so one Claude Code session handles text and
# images: Qwen3.5-9B for chat/tools, Qwen2-VL-2B (+mmproj) for screenshots. The
# server routes each request to the right model (image -> Qwen2-VL, else Qwen3.5)
# and serializes them through one queue. Override any path via the env vars below.
#
# Examples:
#   bash claude_code_launcher.sh                  # text (Qwen3.5) + vision (Qwen2-VL)
#   VISION_MODEL= MMPROJ= bash claude_code_launcher.sh   # text-only (no vision)
#   bash claude_code_launcher.sh /models/other.gguf 11435  # different text model

set -u

# Positional: [model] [port], then an optional `--` and forwarded claude args.
MODEL="/mnt/Data2/DS_projects/llm_models/Qwen3.5-9B-UD-Q6_K_XL.gguf"
PORT="11435"
# Claude Code's context grows fast: a couple of file reads can push the history
# past 50k tokens. A 32k window then can't fit the prompt and the request fails.
# Default to a large window; override with N_CTX=... if VRAM is tight.
N_CTX="${N_CTX:-65536}"
# Multi-GPU split strategy passed to llama_serve_anthropic(). Default "layer"
# splits the model across all GPUs; on a multi-GPU host the Vulkan backend can
# hang doing cross-device copies. Set SPLIT_MODE=none to pin the model to a
# single card (recommended when the model fits in one GPU's VRAM).
SPLIT_MODE="${SPLIT_MODE:-layer}"
# Vision (optional second model): VISION_MODEL is a vision-capable GGUF and
# MMPROJ its paired clip projector. When both are set, image requests route to
# this model; everything else stays on MODEL. Set both empty for a text-only
# server. VISION_N_CTX keeps the vision KV cache small so both fit in VRAM.
VISION_MODEL="${VISION_MODEL:-/mnt/Data2/DS_projects/llm_models/Qwen2-VL-2B-Instruct-Q8_0.gguf}"
MMPROJ="${MMPROJ:-/mnt/Data2/DS_projects/llm_models/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf}"
VISION_N_CTX="${VISION_N_CTX:-8192}"
# Web search/fetch: register the bundled stdio MCP server (mcp_web_search.R) so
# the local model can search the web and read pages. The search runs in that R
# process on this machine (DuckDuckGo, no API key) — nothing hits the network on
# the LLM server side. Set WEB_SEARCH=0 to disable. WEB_SEARCH_SCRIPT overrides
# the path (defaults to the copy next to this launcher).
WEB_SEARCH="${WEB_SEARCH:-1}"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_SEARCH_SCRIPT="${WEB_SEARCH_SCRIPT:-$SELF_DIR/mcp_web_search.R}"
CLAUDE_ARGS=()
[ "${1:-}" != "" ] && [ "${1:-}" != "--" ] && { MODEL="$1"; shift; }
[ "${1:-}" != "" ] && [ "${1:-}" != "--" ] && { PORT="$1"; shift; }
[ "${1:-}" = "--" ] && { shift; CLAUDE_ARGS=("$@"); }

BASE="http://127.0.0.1:${PORT}"
LOG="/tmp/llamar_anthropic_${PORT}.log"
TIMEOUT=300   # large GGUF models can take minutes to load

if [ ! -f "$MODEL" ]; then
  echo "!! model not found: $MODEL" >&2
  exit 1
fi
if ! command -v claude >/dev/null 2>&1; then
  echo "!! 'claude' CLI not found on PATH. Install Claude Code first." >&2
  exit 1
fi

# Refuse to start if the port is already taken. drogonR binds with SO_REUSEPORT,
# so a second server would silently start *alongside* the first one (both
# LISTENing on the same port); the kernel then splits requests between them and
# Claude Code's calls land on whichever — including a stale/old-code server,
# causing erratic failures. Bail out and tell the user what holds the port.
port_listener() {
  # Prints "pid/cmd" of whatever listens on $PORT, if anything (ss, then lsof).
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" | grep -oE 'pid=[0-9]+' | head -1
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1
  fi
}
if [ -n "$(port_listener)" ]; then
  echo "!! port $PORT is already in use:" >&2
  ( ss -ltnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" || lsof -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null ) >&2
  echo "   Another llamaR server (or process) is listening on $PORT." >&2
  echo "   Stop it first, or run this launcher with a different port:" >&2
  echo "     bash $(basename "$0") \"$MODEL\" <other-port>" >&2
  exit 1
fi

echo ">> starting llamaR Anthropic server"
echo "   model: $(basename "$MODEL")"
echo "   port:  $PORT   n_ctx: $N_CTX   split_mode: $SPLIT_MODE   log: $LOG"
# Run the server in its own process group (setsid) so we can signal the whole
# group on cleanup. Rscript forks an inner `exec/R` that actually listens; killing
# only the Rscript wrapper would orphan it, so we target the group instead.
VISION_ARG=""
if [ -n "$VISION_MODEL" ] && [ -n "$MMPROJ" ]; then
  if [ -f "$VISION_MODEL" ] && [ -f "$MMPROJ" ]; then
    VISION_ARG=", vision_model_path='$VISION_MODEL', mmproj_path='$MMPROJ', vision_n_ctx=${VISION_N_CTX}L"
    echo "   vision: $(basename "$VISION_MODEL") (+$(basename "$MMPROJ"), n_ctx=$VISION_N_CTX)"
  else
    echo "   vision: SKIPPED (model or mmproj file missing)" >&2
  fi
fi
setsid Rscript -e "llamaR::llama_serve_anthropic('$MODEL', port=${PORT}L, n_ctx=${N_CTX}L, split_mode='$SPLIT_MODE'${VISION_ARG})" >"$LOG" 2>&1 &
SRV_PID=$!

cleanup() {
  echo
  echo ">> stopping server (pgid $SRV_PID)"
  # SIGINT, not SIGTERM: serve_anthropic()'s serve loop runs inside
  # tryCatch(interrupt=...), so SIGINT unwinds it cleanly (on.exit -> dr_stop()),
  # while SIGTERM is ignored by R and would leave the process running. Send it to
  # the whole group so the inner exec/R (the actual listener) gets it too. If the
  # process is still alive after a grace period, KILL the group as a last resort.
  kill -INT -- "-$SRV_PID" 2>/dev/null
  for _ in 1 2 3 4 5; do
    kill -0 "$SRV_PID" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$SRV_PID" 2>/dev/null
  wait "$SRV_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo ">> waiting for server (up to ${TIMEOUT}s; first load is slow)..."
ready=0
MODELS_JSON=""      # set -u: the loop below may not run if TIMEOUT is 0
for i in $(seq 1 "$TIMEOUT"); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "!! server died before becoming ready. Last log lines:" >&2
    grep -v "radv" "$LOG" | tail -20 >&2
    exit 1
  fi
  if MODELS_JSON="$(curl -s --max-time 2 "${BASE}/v1/models")" && [ -n "$MODELS_JSON" ]; then
    ready=1
    echo "   ready after ${i}s"
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "!! timed out after ${TIMEOUT}s. Last log lines:" >&2
  grep -v "radv" "$LOG" | tail -20 >&2
  exit 1
fi

# Take the model id from the server itself (single source of truth), so the
# name Claude Code displays is the GGUF actually loaded. Fall back to the file
# name if the response cannot be parsed. sed, not jq: no extra dependency.
MODEL_ID="$(printf '%s' "$MODELS_JSON" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -n1)"
[ -n "$MODEL_ID" ] || MODEL_ID="$(basename "$MODEL" .gguf)"
echo ">> launching Claude Code (ANTHROPIC_BASE_URL=$BASE, model=$MODEL_ID)"
echo "   exit claude to stop the server."
echo
# Notes on getting Claude Code to actually use the local server:
#  * Use ONLY ANTHROPIC_AUTH_TOKEN and unset any real ANTHROPIC_API_KEY —
#    having both set makes Claude Code pick the real key and hit
#    api.anthropic.com instead of us.
#  * --model pins the model name sent to the server (the server ignores the
#    name and always serves the loaded GGUF, but this stops Claude Code from
#    requesting "opus").
#  * --settings with inline JSON overrides ~/.claude/settings.json's
#    "model": "opus" without touching the user's global settings.
#  * --mcp-config with inline JSON registers the web-search MCP server for THIS
#    session only (no `claude mcp add`, no change to the user's MCP config);
#    --strict-mcp-config makes Claude Code use ONLY this server, ignoring any
#    globally-configured MCP servers so the session stays self-contained.
MCP_ARGS=()
if [ "$WEB_SEARCH" = "1" ]; then
  if [ -f "$WEB_SEARCH_SCRIPT" ]; then
    echo "   web search: ON ($(basename "$WEB_SEARCH_SCRIPT") via Rscript)"
    MCP_JSON="{\"mcpServers\":{\"web-search\":{\"command\":\"Rscript\",\"args\":[\"$WEB_SEARCH_SCRIPT\"]}}}"
    MCP_ARGS=(--mcp-config "$MCP_JSON" --strict-mcp-config)
  else
    echo "   web search: SKIPPED (script not found: $WEB_SEARCH_SCRIPT)" >&2
  fi
fi
env -u ANTHROPIC_API_KEY \
    ANTHROPIC_BASE_URL="$BASE" \
    ANTHROPIC_AUTH_TOKEN="sk-local" \
  claude --model "$MODEL_ID" \
         --settings "{\"model\":\"$MODEL_ID\"}" \
         "${MCP_ARGS[@]}" \
         "${CLAUDE_ARGS[@]}"
