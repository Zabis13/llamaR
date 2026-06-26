#!/usr/bin/env bash
# Smoke-test llama_serve_anthropic() with curl, without Claude Code.
# Starts the server in the background, waits (up to 5 min for large models to
# load), fires non-stream + stream /v1/messages requests (one with a tool),
# then stops the server.
#
#   bash inst/examples/serve_anthropic_test.sh [model.gguf] [port]

set -u
MODEL="${1:-/mnt/Data2/DS_projects/llm_models/Qwen3-14B-Q8_0.gguf}"
PORT="${2:-11435}"
BASE="http://127.0.0.1:${PORT}"
LOG=/tmp/anthropic_srv.log
TIMEOUT=300   # 5 minutes — large GGUF models can take a while to load from disk

echo ">> starting server (model: $(basename "$MODEL"), port: $PORT)"
Rscript -e "llamaR::llama_serve_anthropic('$MODEL', port=${PORT}L)" >"$LOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null' EXIT

# wait until /v1/models actually answers, the server process dies, or timeout
echo ">> waiting for server (up to ${TIMEOUT}s)..."
ready=0
for i in $(seq 1 "$TIMEOUT"); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "!! server process died before becoming ready. Last log lines:"
    grep -v "radv" "$LOG" | tail -20
    exit 1
  fi
  if curl -s -o /dev/null --max-time 2 "${BASE}/v1/models"; then
    ready=1
    echo "   ready after ${i}s"
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "!! timed out after ${TIMEOUT}s. Last log lines:"
  grep -v "radv" "$LOG" | tail -20
  exit 1
fi

echo
echo "=== GET /v1/models ==="
curl -s "${BASE}/v1/models"; echo

echo
echo "=== POST /v1/messages (non-stream, plain) ==="
curl -s "${BASE}/v1/messages" \
  -H "content-type: application/json" \
  -H "x-api-key: sk-local" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "local",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'; echo

echo
echo "=== POST /v1/messages (non-stream, with tool) ==="
curl -s "${BASE}/v1/messages" \
  -H "content-type: application/json" \
  -d '{
    "model": "local",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "name": "get_weather",
      "description": "Get current weather for a city",
      "input_schema": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
    }]
  }'; echo

echo
echo "=== POST /v1/messages (stream, with tool) — raw SSE frames ==="
curl -s -N "${BASE}/v1/messages" \
  -H "content-type: application/json" \
  -d '{
    "model": "local",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "name": "get_weather",
      "description": "Get current weather for a city",
      "input_schema": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
    }]
  }'; echo

echo
echo ">> done (stopping server)"
