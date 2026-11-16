#!/usr/bin/env bash
# crash_bt.sh - attach gdb to the running strata-poly engine and print the backtrace of its next crash.
#
#   ./crash_bt.sh              # wait for the engine, attach, print the backtrace at the crash
#   ./crash_bt.sh --loop       # the same, then re-attach to the restarted engine for the next crash
#   ./crash_bt.sh 12345        # attach to that pid
#   ENGINE_PAT='strata-poly --serve' ./crash_bt.sh
#
# While gdb is attached the engine answers as usual: send the request that kills it and read the
# backtrace here and in crash_bt.log.
set -u

PAT="${ENGINE_PAT:-strata-poly --serve}"
LOOP=0
PID=""
for a in "$@"; do
    [[ "$a" == "--loop" ]] && LOOP=1 || PID="$a"
done

find_pid() {
    pgrep -f "$PAT" | while read -r p; do ps -o etimes= -p "$p" 2>/dev/null | grep -q . && echo "$p"; done | tail -1
}

echo "ищу движок по '$PAT' — подними сервер, если он ещё не запущен (скрипт ждёт, Ctrl-C — выйти)"
PID="${PID:-$(find_pid)}"
WAITED=0
while [[ -z "$PID" ]]; do
    sleep 5
    WAITED=$((WAITED + 5))
    echo -ne "   жду движок: ${WAITED}s\r"
    PID="$(find_pid)"
done
echo ""

attach() {
    echo "attach к $PID; отправь в чат запрос, который роняет движок, — backtrace будет здесь и в $PWD/crash_bt.log"
    gdb -p "$1" -batch \
        -ex continue \
        -ex "echo \n===== BACKTRACE (crash) =====\n" \
        -ex bt \
        -ex "echo \n===== ВСЕ ПОТОКИ =====\n" \
        -ex "thread apply all bt" \
        2>&1 | tee crash_bt.log
}

attach "$PID"
while [[ "$LOOP" == 1 ]]; do
    echo "движок перезапустится сам; жду новый процесс (Ctrl-C — выйти)"
    PID=""
    while [[ -z "$PID" ]]; do
        sleep 5
        PID="$(find_pid)"
    done
    attach "$PID"
done
echo "готово: $PWD/crash_bt.log"
