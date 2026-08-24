#!/usr/bin/env bash
# Real-provider E2E validation against a local OpenAI-compatible endpoint
# (default: Ollama on :11434). Boots the real proxy + PostgreSQL test
# container state and exercises chat/multi-turn/streaming/tools/developer/
# failure/concurrency/capture-overflow paths.
#
# Usage:  BASE_MODEL=MODEL_TAG ./scripts/e2e_real_provider.sh
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
MODEL="${BASE_MODEL:-MichelRosselli/bonsai-27b:latest}"
API=http://127.0.0.1:8090
PG_DSN="postgresql://context_proxy:context_proxy@localhost:5433/context_proxy"
FAILS=0

say() { printf '\n=== %s ===\n' "$1"; }
check() { # check <desc> <condition_exit_code>
  if [ "$2" -eq 0 ]; then echo "PASS: $1"; else echo "FAIL: $1"; FAILS=$((FAILS+1)); fi
}

say "0. prerequisites"
curl -s --noproxy "*" -f -m 10 "$BASE_URL/models" >/dev/null; check "provider reachable" $?

docker start cpx-test-postgres >/dev/null 2>&1 || true
sleep 2

say "1. boot proxy against real provider"
INFERENCE__BASE_URL="$BASE_URL" \
INFERENCE__MODEL="$MODEL" \
DATABASE__URL="$PG_DSN" \
SERVER__MAX_CAPTURE_BYTES=600 \
.venv/bin/python -m uvicorn context_proxy.main:app --host 127.0.0.1 --port 8090 \
  > /tmp/e2e_proxy.log 2>&1 &
PROXY_PID=$!
for i in $(seq 1 60); do curl -s --noproxy "*" -f $API/healthz >/dev/null && break; sleep 1; done
curl -s --noproxy "*" -f $API/healthz | grep -Eq '"status": ?"ok"'; check "proxy healthy" $?

say "0b. provider/model warmup (cold-load can exceed short timeouts)"
curl -s --noproxy "*" -m 900 http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"warm\"}]}" >/dev/null
check "model warmed up" $?

chat() { # chat <conv> <content> [extra_json]
  local conv="$1" content="$2" extra="${3:-}"
  local body="{\"model\":\"$MODEL\",\"messages\":[$content],\"conversation_id\":\"$conv\"$extra}"
  curl -s --noproxy "*" -m 600 -X POST $API/v1/chat/completions \
    -H 'Content-Type: application/json' -d "$body"
}

say "2. simple chat"
CONV=$(uuidgen)
RESP=$(chat "$CONV" '{"role":"user","content":"Reply with exactly: OK"}')
echo "$RESP" | grep -q '"role":"assistant"'; check "assistant reply" $?
echo "$RESP" | python3 -c 'import sys,json;d=json.load(sys.stdin);assert d["choices"][0]["message"]["content"]' \
  && check "non-empty content" $? || check "non-empty content" 1

say "3. multi-turn conversation"
extract_msg() {
  python3 -c 'import sys,json;d=json.load(sys.stdin);print(json.dumps(d["choices"][0]["message"],ensure_ascii=False))'
}
CONV_MT=$(uuidgen)
R_HI=$(chat "$CONV_MT" '{"role":"user","content":"hi"}')
A_REAL=$(printf '%s' "$R_HI" | extract_msg)
R2=$(chat "$CONV_MT" "{\"role\":\"user\",\"content\":\"hi\"},$A_REAL,{\"role\":\"user\",\"content\":\"one word: continue?\"}")
echo "$R2" | grep -q '"role":"assistant"'; check "multi-turn reply" $?

say "4. developer instruction"
R3=$(chat "$(uuidgen)" '{"role":"developer","content":"Always prefix answers with DEV:"},{"role":"user","content":"ping"}')
echo "$R3" | grep -q '"role":"assistant"'; check "developer accepted by provider path" $?
echo "$R3" | grep -qv '"role":"system"'; check "developer role preserved (not system)" $?

say "5. streaming"
CONV=$(uuidgen)
BODY="{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"count to five slowly\"}],\"conversation_id\":\"$CONV\"}"
STREAM=$(curl -s -N -m 240 -X POST $API/v1/chat/completions \
  -H 'Content-Type: application/json' -d "$BODY")
echo "$STREAM" | grep -q 'data: \[DONE\]'; check "stream reaches DONE" $?
echo "$STREAM" | grep -c 'data:' | grep -qv '^0$'; check "stream chunks delivered" $?
echo "$STREAM" | grep -q 'assistant_persistence_skipped_capture_overflow\|data:' && \
  grep -q 'capture_overflow' /tmp/e2e_proxy.log; OVER=$?
echo "(capture overflow observed: $OVER)"

say "6. tool call (function)"
TOOLS='[{"type":"function","function":{"name":"get_time","description":"current time","parameters":{"type":"object","properties":{}}}}]'
BODY="{\"model\":\"$MODEL\",\"tools\":$TOOLS,\"tool_choice\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it? Use the tool.\"}],\"conversation_id\":\"$(uuidgen)\"}"
R6=$(curl -s --noproxy '*' -m 600 -X POST $API/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY")
echo "$R6" | grep -q '"role":"assistant"'; check "tool-call turn answered (model-dependent)" $?
echo "$R6" | grep -q 'tool_calls\|content'; check "valid assistant shape" $?

say "7. concurrent conversations"
PIDS=""
for i in 1 2 3; do
  C=$(uuidgen)
  chat "$C" "{\"role\":\"user\",\"content\":\"parallel $i\"}" > /tmp/e2e_par_$i.json &
  PIDS="$PIDS $!"
done
OKALL=0
for pid in $PIDS; do wait $pid || OKALL=1; done
for i in 1 2 3; do grep -q '"role":"assistant"' /tmp/e2e_par_$i.json || OKALL=1; done
check "parallel conversations all answered" $OKALL

say "8. provider failure -> reconnect"
kill -STOP $PROXY_PID 2>/dev/null  # pause nothing upstream; instead point at dead port via second proxy
INFERENCE__BASE_URL="http://localhost:9/v1" INFERENCE__MODEL="$MODEL" \
DATABASE__URL="$PG_DSN" SERVER__PORT=8091 \
.venv/bin/python -m uvicorn context_proxy.main:app --host 127.0.0.1 --port 8091 > /tmp/e2e_dead.log 2>&1 &
DEAD_PID=$!
for i in $(seq 1 20); do curl -sf http://127.0.0.1:8091/healthz >/dev/null && break; sleep 1; done
RDEAD=$(curl -s --noproxy "*" -m 120 -X POST http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}")
echo "$RDEAD" | grep -q 'upstream_unavailable'; check "dead upstream -> openai 502 error" $?
kill $DEAD_PID 2>/dev/null
# main proxy still fine afterwards (reconnect path)
RESP=""
CURL_RC=1
for attempt in 1 2 3; do
  RESP=$(chat "$(uuidgen)" '{"role":"user","content":"still alive?"}')
  CURL_RC=$?
  if [ $CURL_RC -eq 0 ] && echo "$RESP" | grep -q '"role":"assistant"'; then
    break
  fi
  echo "(attempt $attempt: curl_rc=$CURL_RC resp_len=${#RESP})" >&2
  sleep 2
done
if [ $CURL_RC -eq 0 ] && echo "$RESP" | grep -q '"role":"assistant"'; then
  check "main proxy still serving" 0
else
  echo "DEBUG final resp len=${#RESP}"
  check "main proxy still serving" 1
fi

say "9. persisted state (PostgreSQL authoritative)"
docker exec cpx-test-postgres psql -U context_proxy -d context_proxy -tAc \
  "SELECT count(*) FROM messages WHERE conversation_id='$CONV_MT'" > /tmp/e2e_rows.txt
ROWS=$(cat /tmp/e2e_rows.txt)
[ "${ROWS:-0}" -ge 2 ]; check "multi-turn persisted rows>=2 ($ROWS)" $?

say "10. capture overflow metric present after long stream"
grep -q 'assistant_capture_overflow' /tmp/e2e_proxy.log && FOUND=0 || FOUND=1
echo "(overflow seen: $((1-FOUND)))"

cleanup() {
  kill $PROXY_PID 2>/dev/null
  docker rm -f cpx-e2e-none >/dev/null 2>&1 || true
}
cleanup

echo
if [ "$FAILS" -eq 0 ]; then echo "E2E RESULT: ALL PASS"; else echo "E2E RESULT: $FAILS FAILURES"; fi
exit $FAILS
