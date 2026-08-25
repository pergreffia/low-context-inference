#!/usr/bin/env bash
# Real-provider E2E validation against a local OpenAI-compatible endpoint
# (default: Ollama on :11434). Boots the real proxy + PostgreSQL test
# container and exercises chat/multi-turn/streaming/tools/developer/
# failure-recovery/concurrency paths.
#
# Failure policy: aggregate reporting WITHOUT `set -e` — therefore every
# checked operation goes through run_check (explicit failure handling) and an
# unconditional EXIT trap guarantees cleanup. Unhandled command failures in a
# section are caught by the explicit rc propagation below; the final exit code
# is non-zero whenever any required check failed. Model-dependent outcomes are
# reported as SKIP, never PASS.
#
# NOTE (authoritative vs best-effort): capture-overflow, developer-role
# persistence/context and tool-call compatibility have DETERMINISTIC
# fake-provider regression tests in tests/ (see tests/test_capture_overflow.py,
# tests/test_developer_contract.py, tests/test_tool_lifecycle.py). The checks
# here are complementary live-provider validation; the ones marked
# model-dependent are informational only.
#
# Usage:  BASE_MODEL=MODEL_TAG ./scripts/e2e_real_provider.sh
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
MODEL="${BASE_MODEL:-MichelRosselli/bonsai-27b:latest}"
API=http://127.0.0.1:8090
PG_DSN="postgresql://context_proxy:context_proxy@localhost:5433/context_proxy"
FWD_PORT=8123   # controllable upstream in front of the real provider
FAILS=0
SKIPS=0

PROXY_PID=""
FWD_PID=""
TMP_FILES=()

say() { printf '\n=== %s ===\n' "$1"; }

run_check() { # run_check <description> <command...>
    local description="$1"
    shift
    if "$@"; then
        echo "PASS: $description"
        return 0
    fi
    local rc=$?
    echo "FAIL: $description"
    FAILS=$((FAILS + 1))
    return "$rc"
}

skip() { # skip <description> <reason>
    echo "SKIP: $1 ($2)"
    SKIPS=$((SKIPS + 1))
}

info() { echo "INFO: $1"; }

note_tmp() { TMP_FILES+=("$1"); }

cleanup() {
    # Best-effort, error-suppressed: cleanup must never overwrite the real
    # test result carried by $?/FAILS.
    [ -n "$PROXY_PID" ] && kill "$PROXY_PID" 2>/dev/null
    [ -n "$FWD_PID" ] && kill "$FWD_PID" 2>/dev/null
    wait 2>/dev/null
    local f
    for f in ${TMP_FILES[@]+"${TMP_FILES[@]}"}; do rm -rf "$f" 2>/dev/null || true; done
}
trap cleanup EXIT

curl_api() { # curl_api <curl args...>  (never uses ambient proxies)
    curl -s --noproxy "*" "$@"
}

pg_query() { # pg_query <sql>
    docker exec cpx-test-postgres psql -U context_proxy -d context_proxy -tAc "$1" 2>/dev/null
}

pg_ready() {
    [ "$(pg_query 'SELECT 1')" = "1" ]
}

wait_healthy() { # wait_healthy <url> <tries>
    local url="$1" tries="$2" i
    for i in $(seq 1 "$tries"); do
        curl_api -f -m 5 "$url" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

upstream_via_fwd() { # poll until the real provider answers through the forwarder
    local i
    for i in $(seq 1 30); do
        curl_api -f -m 15 "http://127.0.0.1:$FWD_PORT/v1/models" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

chat() { # chat <conv> <messages_json_array> [extra_json]
    local conv="$1" messages="$2" extra="${3:-}"
    local body="{\"model\":\"$MODEL\",\"messages\":[$messages],\"conversation_id\":\"$conv\"$extra}"
    curl_api -m 600 -X POST $API/v1/chat/completions \
        -H 'Content-Type: application/json' -d "$body"
}

extract_msg() {
    python3 -c 'import sys,json;d=json.load(sys.stdin);print(json.dumps(d["choices"][0]["message"],ensure_ascii=False))'
}

# Controllable TCP forwarder: proxy -> forwarder -> real provider.
# Killing/restarting it makes "provider unavailable / recovered"
# deterministic while the SAME proxy process keeps running.
# Both sockets are closed once BOTH directions finish (no fd leaks).
start_forwarder() {
    python3 -c "
import socket, threading
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(('127.0.0.1', $FWD_PORT))
listener.listen(64)
UP = ('127.0.0.1', 11434)

def pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try: dst.shutdown(socket.SHUT_WR)
        except OSError: pass

def relay(a, b):
    other = threading.Thread(target=pump, args=(a, b), daemon=True)
    other.start()
    pump(b, a)
    other.join()
    for s in (a, b):
        try: s.close()
        except OSError: pass

while True:
    client, _ = listener.accept()
    try:
        up = socket.create_connection(UP, timeout=10)
        # create_connection's timeout STAYS on the socket: without this,
        # an idle recv() (slow model prompt-processing) aborts the relay
        # after 10s and every proxied request fails as upstream_unavailable.
        up.settimeout(None)
    except OSError:
        client.close()
        continue
    threading.Thread(target=relay, args=(client, up), daemon=True).start()
" >"$E2E_TMPDIR/forwarder.log" 2>&1 &
    FWD_PID=$!
}

say "0. prerequisites"
provider_reachable() {
    curl_api -f -m 10 "$BASE_URL/models" >/dev/null 2>&1
}
run_check "provider reachable" provider_reachable

docker start cpx-test-postgres >/dev/null 2>&1 || true
sleep 2
run_check "postgres container ready" pg_ready

E2E_TMPDIR="$(mktemp -d /tmp/e2e_rp.XXXXXX)"
note_tmp "$E2E_TMPDIR"

say "1. boot proxy against real provider (via controllable upstream)"
# Breaker knobs: generous threshold so ONE flaky upstream timeout does not
# wedge the whole run, short reset window so recovery is observable quickly.
INFERENCE__BASE_URL="http://127.0.0.1:$FWD_PORT/v1" \
INFERENCE__MODEL="$MODEL" \
DATABASE__URL="$PG_DSN" \
SERVER__MAX_CAPTURE_BYTES=600 \
RESILIENCE__BREAKER_FAILURE_THRESHOLD=8 \
RESILIENCE__BREAKER_RESET_SECONDS=3 \
.venv/bin/python -m uvicorn context_proxy.main:app --host 127.0.0.1 --port 8090 \
    > "$E2E_TMPDIR/proxy.log" 2>&1 &
PROXY_PID=$!
PROXY_LOG="$E2E_TMPDIR/proxy.log"

start_forwarder
run_check "upstream path through forwarder" upstream_via_fwd

if run_check "proxy healthy" wait_healthy "$API/healthz" 60; then
    :
else
    echo "proxy never became healthy; log tail:" >&2
    tail -20 "$E2E_TMPDIR/proxy.log" >&2
    exit 1
fi

say "0b. provider/model warmup (cold-load can exceed short timeouts)"
warm_up_model() {
    curl_api -f -m 900 http://127.0.0.1:8090/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"warm\"}]}" \
        >/dev/null
}
WARM_OK=1
for attempt in 1 2 3; do
    if warm_up_model; then WARM_OK=0; break; fi
    echo "(warmup attempt $attempt failed; retrying)" >&2
    sleep 2
done
if [ "$WARM_OK" -eq 0 ]; then
    echo "PASS: model warmed up"
else
    echo "FAIL: model warmed up"; FAILS=$((FAILS + 1))
fi

say "2. simple chat"
CONV=$(uuidgen)
RESP=$(chat "$CONV" '{"role":"user","content":"Reply with exactly: OK"}')
if echo "$RESP" | grep -q '"role":"assistant"'; then
    echo "PASS: assistant reply"
else
    echo "FAIL: assistant reply"; FAILS=$((FAILS + 1))
fi
if echo "$RESP" | python3 -c 'import sys,json;d=json.load(sys.stdin);assert d["choices"][0]["message"]["content"]'; then
    echo "PASS: non-empty content"
else
    echo "FAIL: non-empty content"; FAILS=$((FAILS + 1))
fi

say "3. multi-turn conversation"
CONV_MT=$(uuidgen)
R_HI=$(chat "$CONV_MT" '{"role":"user","content":"hi"}')
if printf '%s' "$R_HI" | grep -q '"choices"'; then
    A_REAL=$(printf '%s' "$R_HI" | extract_msg)
    R2=$(chat "$CONV_MT" "{\"role\":\"user\",\"content\":\"hi\"},$A_REAL,{\"role\":\"user\",\"content\":\"one word: continue?\"}")
    if echo "$R2" | grep -q '"role":"assistant"'; then
        echo "PASS: multi-turn reply"
    else
        echo "FAIL: multi-turn reply (body=$(printf '%s' "$R2" | head -c 200))"; FAILS=$((FAILS + 1))
    fi
else
    echo "FAIL: multi-turn first turn (body=$(printf '%s' "$R_HI" | head -c 200))"; FAILS=$((FAILS + 1))
fi

say "4. developer instruction (Test A: OpenAI-compat acceptance)"
CONV_DEV=$(uuidgen)
DEV_MSG='{"role":"developer","content":"Always prefix answers with DEV:"}'
R3=$(chat "$CONV_DEV" "$DEV_MSG,{\"role\":\"user\",\"content\":\"ping\"}")
if echo "$R3" | grep -q '"role":"assistant"'; then
    echo "PASS: developer accepted by provider path"
else
    echo "FAIL: developer accepted by provider path"; FAILS=$((FAILS + 1))
fi
# Persistence/context contract spot-check against PostgreSQL (authoritative
# coverage lives in tests/test_developer_contract.py).
DEV_ROLE_ROWS=$(pg_query "SELECT count(*) FROM messages WHERE conversation_id='$CONV_DEV' AND role='developer'")
if [ "${DEV_ROLE_ROWS:-0}" -ge 1 ]; then
    echo "PASS: developer role persisted verbatim ($DEV_ROLE_ROWS row)"
else
    echo "FAIL: developer role persisted verbatim (rows=${DEV_ROLE_ROWS:-<none>})"; FAILS=$((FAILS + 1))
fi

say "5. streaming"
CONV=$(uuidgen)
BODY="{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"count to five slowly\"}],\"conversation_id\":\"$CONV\"}"
curl_api -N -m 240 -X POST $API/v1/chat/completions \
    -H 'Content-Type: application/json' -d "$BODY" > "$E2E_TMPDIR/stream.sse"
if grep -q 'data: \[DONE\]' "$E2E_TMPDIR/stream.sse"; then
    echo "PASS: stream reaches DONE"
else
    echo "FAIL: stream reaches DONE"; FAILS=$((FAILS + 1))
fi
if grep -q 'data:' "$E2E_TMPDIR/stream.sse"; then
    echo "PASS: stream chunks delivered"
else
    echo "FAIL: stream chunks delivered"; FAILS=$((FAILS + 1))
fi
# Capture overflow after a long stream is MODEL-DEPENDENT here (the response
# may or may not exceed SERVER__MAX_CAPTURE_BYTES=600). Informational only;
# the authoritative deterministic test is tests/test_capture_overflow.py.
if grep -q 'capture_overflow' "$PROXY_LOG"; then
    info "capture overflow observed on long stream (best-effort scenario hit)"
else
    skip "capture overflow on live stream" "response stayed under capture bound (expected possible); deterministic coverage in tests/"
fi

say "6. tool call (function, model-dependent)"
TOOLS='[{"type":"function","function":{"name":"get_time","description":"current time","parameters":{"type":"object","properties":{}}}}]'
BODY="{\"model\":\"$MODEL\",\"tools\":$TOOLS,\"tool_choice\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"What time is it? Use the tool.\"}],\"conversation_id\":\"$(uuidgen)\"}"
R6=$(curl_api -m 600 -X POST $API/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY")
if echo "$R6" | grep -q '"tool_calls"'; then
    echo "PASS: provider emitted function tool_call through proxy"
elif echo "$R6" | grep -q '"role":"assistant"'; then
    skip "function tool-call compatibility" "model did not choose tool (deterministic coverage in tests/test_tool_lifecycle.py)"
else
    echo "FAIL: tool turn produced no usable assistant response"; FAILS=$((FAILS + 1))
fi

say "7. concurrent conversations"
PIDS=""
for i in 1 2 3; do
    C=$(uuidgen)
    chat "$C" "{\"role\":\"user\",\"content\":\"parallel $i\"}" > "$E2E_TMPDIR/par_$i.json" &
    PIDS="$PIDS $!"
done
OKALL=0
FAILED_SLOTS=""
for pid in $PIDS; do wait "$pid" || OKALL=1; done
for i in 1 2 3; do
    if ! grep -q '"role":"assistant"' "$E2E_TMPDIR/par_$i.json"; then
        FAILED_SLOTS="$FAILED_SLOTS $i"
        OKALL=1
    fi
done
# One sequential retry round for failed slots (local models can legitimately
# shed parallel load); persistent capacity errors are SKIP, not PASS.
CAPACITY_ONLY=0
for i in $FAILED_SLOTS; do
    RESP_RETRY=$(chat "$(uuidgen)" "{\"role\":\"user\",\"content\":\"parallel retry $i\"}")
    if echo "$RESP_RETRY" | grep -q '"role":"assistant"'; then
        echo "$RESP_RETRY" > "$E2E_TMPDIR/par_$i.json"
    elif printf '%s' "$RESP_RETRY" | grep -Eq 'overloaded|rate.limit|capacity|503'; then
        CAPACITY_ONLY=1
    else
        CAPACITY_ONLY=0
        break
    fi
done
if [ "$OKALL" -eq 0 ]; then
    echo "PASS: parallel conversations all answered"
elif [ "$CAPACITY_ONLY" -eq 1 ]; then
    skip "parallel conversations all answered" "provider shed concurrent load even after retry (deterministic concurrency coverage in tests/)"
else
    echo "FAIL: parallel conversations all answered"; FAILS=$((FAILS + 1))
fi

say "8. provider failure -> recovery THROUGH THE SAME PROXY"
# Deterministic: stop the forwarder (provider unavailable), expect the
# contract failure, restart it, expect recovery — same proxy process, same port.
kill "$FWD_PID" 2>/dev/null
wait "$FWD_PID" 2>/dev/null
FWD_PID=""
DEAD_STATUS=$(curl_api -m 120 -o "$E2E_TMPDIR/dead.json" -w '%{http_code}' \
    -X POST $API/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}")
if [ "${DEAD_STATUS:-000}" = "502" ] && grep -q 'upstream_unavailable' "$E2E_TMPDIR/dead.json"; then
    echo "PASS: dead upstream -> openai 502 error"
else
    echo "FAIL: dead upstream -> openai 502 error (status=${DEAD_STATUS:-<none>} body=$(head -c 200 "$E2E_TMPDIR/dead.json" 2>/dev/null))"; FAILS=$((FAILS + 1))
fi
start_forwarder
# The outage may have tripped the breaker; wait (bounded) for it to allow
# probes again — same proxy process throughout.
BREAKER_OPEN=1
for i in $(seq 1 20); do
    if curl_api -f -m 5 $API/readyz | grep -Eq '"circuit_breaker": ?"(closed|half_open)"'; then
        BREAKER_OPEN=0
        break
    fi
    sleep 1
done
[ "$BREAKER_OPEN" -eq 1 ] && echo "(breaker still open after ${i}s; continuing with retries)" >&2
RECOVERED=1
for attempt in 1 2 3 4 5; do
    RESP=$(chat "$(uuidgen)" '{"role":"user","content":"still alive?"}')
    CURL_RC=$?
    if [ "$CURL_RC" -eq 0 ] && echo "$RESP" | grep -q '"role":"assistant"'; then
        RECOVERED=0
        break
    fi
    echo "(attempt $attempt: curl_rc=$CURL_RC resp_len=${#RESP})" >&2
    sleep 2
done
if [ "$RECOVERED" -eq 0 ]; then
    echo "PASS: SAME proxy instance serving again after provider recovery"
else
    echo "FAIL: SAME proxy instance serving again after provider recovery"; FAILS=$((FAILS + 1))
fi

say "9. persisted state (PostgreSQL authoritative)"
ROWS=$(pg_query "SELECT count(*) FROM messages WHERE conversation_id='$CONV_MT'")
if [ "${ROWS:-0}" -ge 2 ]; then
    echo "PASS: multi-turn persisted rows>=2 ($ROWS)"
else
    echo "FAIL: multi-turn persisted rows>=2 (${ROWS:-<none>})"; FAILS=$((FAILS + 1))
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "E2E RESULT: ALL REQUIRED CHECKS PASS ($SKIPS skipped)"
else
    echo "E2E RESULT: $FAILS FAILURES ($SKIPS skipped)"
fi
exit "$FAILS"
