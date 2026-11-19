#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""serve_dialog_logger.py - логирующий прокси перед движком: пересылает запросы на
127.0.0.1:8080 как есть (стриминг SSE проходит насквозь), а каждый завершённый
chat/completions дописывает в ~/eagle_gen/dialogs.jsonl: {"ts", "messages", "answer"}.
Это доменные данные для ft7: метки = пики самой модели на реальных вопросах.

    python3 serve_dialog_logger.py [порт]      # default 8090
    # в UI: адрес http://<host>:8090, ключ тот же; движок живёт на 8080 как обычно
"""
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP_HOST = os.environ.get("UP_HOST", "127.0.0.1")
UP_PORT = int(os.environ.get("UP_PORT", "8080"))
LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
OUT = os.path.expanduser("~/eagle_gen/dialogs.jsonl")
LOCK = threading.Lock()

HOP = {"host", "content-length", "connection", "transfer-encoding", "keep-alive"}


def save(req, answer):
    rec = {"ts": time.time(), "answer": answer}
    if req.get("messages"):
        rec["messages"] = req["messages"]
    if req.get("prompt"):
        rec["prompt"] = req["prompt"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with LOCK:
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract(req, chunks, stream):
    raw = b"".join(chunks).decode("utf-8", "replace")
    if not stream:
        try:
            return json.loads(raw)["choices"][0]["message"]["content"]
        except Exception:
            return ""
    out = []
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            d = json.loads(line[6:])
            delta = d["choices"][0].get("delta") or {}
            out.append(delta.get("content") or "")
        except Exception:
            pass
    return "".join(out)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _forward(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        req = None
        try:
            req = json.loads(body)
        except Exception:
            pass
        try:
            c = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=600)
            c.request(self.command, self.path, body=body,
                      headers={k: v for k, v in self.headers.items() if k.lower() not in HOP})
            r = c.getresponse()
        except Exception as e:
            self.send_error(502, f"upstream: {e!r}")
            return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() in HOP:
                continue
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.close_connection = True
        stream = bool(req and req.get("stream"))
        chunks = []
        if stream:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                piece = r.read(4096)
                if not piece:
                    break
                chunks.append(piece)
                self.wfile.write(("%x\r\n" % len(piece)).encode() + piece + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            data = r.read()
            chunks.append(data)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        c.close()
        if req and (req.get("messages") or req.get("prompt")) and r.status == 200:
            try:
                save(req, extract(req, chunks, stream))
            except Exception as e:
                sys.stderr.write("save: %r\n" % (e,))

    do_POST = _forward
    do_GET = _forward
    do_PUT = _forward
    do_DELETE = _forward


if __name__ == "__main__":
    print(f"dialog logger: listen :{LISTEN} -> {UP_HOST}:{UP_PORT}, out {OUT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", LISTEN), H).serve_forever()
