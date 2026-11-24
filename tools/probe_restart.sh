#!/usr/bin/env bash
# Рестарт сервера (как это делает свип) и сразу проба качества — правка конфига применяется гарантированно.
#
#   ./tools/probe_restart.sh --long
#   ./tools/probe_restart.sh            # без длинной проверки, ~40 с
#
# Останавливает текущий сервер, поднимает новый тем же способом, что bench_sweep.sh (serve/server.py,
# порт и хост из конфига), ждёт /health с "loaded": true и запускает tools/quality_probe.py.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
CFG="${CFG:-strata-dsv4-ud-iq2_xxs.json}"

python3 - "$ROOT/$CFG" <<'PY' || exit 1
import http.client, json, os, subprocess, sys, time

cfg_path = sys.argv[1]
cfg = json.load(open(cfg_path, encoding="utf-8-sig"))
host = str(cfg.get("host") or "").strip() or "127.0.0.1"
if host in ("0.0.0.0", "::"):
    host = "127.0.0.1"
port = int(cfg.get("port") or 1524)
key = cfg.get("api_key") or ""
root = os.path.dirname(os.path.dirname(os.path.abspath(cfg_path)))

def alive():
    try:
        c = http.client.HTTPConnection(host, port, timeout=2)
        c.connect(); c.close()
        return True
    except OSError:
        return False

if alive():
    print("  останавливаю сервер ...", flush=True)
    subprocess.run(["pkill", "-TERM", "-f", f"{root}/serve/server.py"], check=False)
    for _ in range(90):
        if not alive():
            break
        time.sleep(1)
    subprocess.run(["pkill", "-KILL", "-f", f"{root}/serve/server.py"], check=False)
    subprocess.run(["pkill", "-KILL", "-f", "strata-poly"], check=False)
    for _ in range(30):
        if not subprocess.run(["pgrep", "-f", "strata-poly"], capture_output=True).stdout:
            break
        time.sleep(1)
    time.sleep(2)

log = open("/tmp/probe-server.log", "w", encoding="utf-8")
subprocess.Popen([f"{root}/.venv/bin/python", f"{root}/serve/server.py", "--engine", "strata",
                  "--config", cfg_path, "--port", str(port)],
                 cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ})
print(f"  сервер поднимается (лог /tmp/probe-server.log), жду /health ...", flush=True)
for _ in range(1200):          # модель грузится до 5 минут, но дадим запас
    try:
        c = http.client.HTTPConnection(host, port, timeout=5)
        c.request("GET", "/health", headers={"Authorization": "Bearer " + key})
        body = c.getresponse().read().decode("utf-8", "replace")
        if '"loaded"' in body and "true" in body:
            print("  сервер готов", flush=True)
            break
    except OSError:
        pass
    time.sleep(3)
else:
    print("!!! сервер не поднялся: смотри /tmp/probe-server.log", flush=True)
    sys.exit(1)
PY

exec python3 tools/quality_probe.py "$@"
