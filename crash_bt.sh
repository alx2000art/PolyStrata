#!/usr/bin/env bash
# crash_bt.sh - attach gdb to the running strata-poly engine and print the backtrace of its next crash.
#
#   ./crash_bt.sh                  # wait for the engine, attach, print the backtrace at the crash
#   ./crash_bt.sh 12345            # attach to that pid
#   ENGINE_PAT='strata-poly --serve' ./crash_bt.sh
#
# The server restarts the engine after a crash - run the script again for the next one.  While gdb is
# attached the engine answers as usual; send the request that kills it and read the backtrace.
set -u

PAT="${ENGINE_PAT:-strata-poly --serve}"
PID="${1:-}"

if [[ -z "$PID" ]]; then
    echo "жду движок ($PAT)..."
    for _ in $(seq 1 180); do
        PID=$(pgrep -f "$PAT" | head -1)
        [[ -n "$PID" ]] && break
        sleep 1
    done
fi
[[ -n "$PID" ]] || { echo "движок не запущен — подними сервер и повтори"; exit 1; }

echo "attach к $PID; отправь в чат запрос, который роняет движок, — backtrace будет здесь и в $PWD/crash_bt.log"
gdb -p "$PID" -batch \
    -ex continue \
    -ex "echo \n===== BACKTRACE (crash) =====\n" \
    -ex bt \
    -ex "echo \n===== ВСЕ ПОТОКИ =====\n" \
    -ex "thread apply all bt" \
    2>&1 | tee crash_bt.log
echo "готово: $PWD/crash_bt.log (движок после краша перезапустится сам)"
