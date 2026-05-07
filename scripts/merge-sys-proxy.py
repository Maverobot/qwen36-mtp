#!/usr/bin/env python3
"""
Tiny streaming HTTP proxy that merges consecutive `system` messages in
OpenAI chat completions requests. Needed because the Qwen3 chat template
embedded in the GGUF refuses any system message that isn't at index 0.

Usage:
  qwen36-merge-sys-proxy.py LISTEN_PORT UPSTREAM_URL

Forwards everything else verbatim. Streaming responses are passed through
chunk-by-chunk so SSE works. Designed to sit between an agent harness
(omp/opencode) and llama-server.
"""
import json
import socketserver
import sys
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler


def merge_system_messages(payload):
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return payload
    sys_parts = []
    other = []
    for m in msgs:
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str):
                sys_parts.append(c)
            elif isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and "text" in item:
                        sys_parts.append(item["text"])
                    elif isinstance(item, str):
                        sys_parts.append(item)
        else:
            other.append(m)
    if sys_parts:
        merged = [{"role": "system", "content": "\n\n".join(sys_parts)}]
    else:
        merged = []
    payload["messages"] = merged + other

    # Disable Qwen3 "thinking" mode by default. The Qwen3 chat template wraps
    # the assistant turn in <think>...</think> unless `enable_thinking` is
    # explicitly set to false. Most coding harnesses (omp, opencode) don't
    # know how to surface reasoning_content, so they appear to hang while the
    # model writes a long internal chain of thought. Allow callers to
    # re-enable thinking by sending their own chat_template_kwargs.
    ctk = payload.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        ctk = {}
        payload["chat_template_kwargs"] = ctk
    ctk.setdefault("enable_thinking", False)
    return payload


def make_handler(upstream):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _forward(self, method):
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else None
            if body and self.path.endswith("/chat/completions"):
                try:
                    payload = json.loads(body)
                    payload = merge_system_messages(payload)
                    body = json.dumps(payload).encode()
                except Exception:
                    pass
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
            if body is not None:
                headers["Content-Length"] = str(len(body))
            req = urllib.request.Request(upstream + self.path, data=body, headers=headers, method=method)
            try:
                resp = urllib.request.urlopen(req, timeout=600)
            except urllib.error.HTTPError as e:
                resp = e
            self.send_response(resp.status if hasattr(resp, "status") else resp.code)
            is_stream = False
            content_length = None
            for k, v in resp.headers.items():
                lk = k.lower()
                if lk in ("transfer-encoding", "connection", "content-length"):
                    if lk == "content-length":
                        content_length = v
                    continue
                if lk == "content-type" and "event-stream" in v.lower():
                    is_stream = True
                self.send_header(k, v)
            self.send_header("Connection", "close")
            if is_stream or content_length is None:
                # Stream with chunked transfer; close connection at EOF so the
                # client (e.g. opencode's fetch) knows the response ended.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass
            else:
                # Buffered, length-known response: forward as-is with
                # Content-Length so HTTP/1.1 framing is unambiguous.
                data = resp.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

        def do_POST(self):
            self._forward("POST")

        def do_GET(self):
            self._forward("GET")

        def log_message(self, *args, **kwargs):
            pass

    return Proxy


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    if len(sys.argv) != 3:
        print("usage: qwen36-merge-sys-proxy.py LISTEN_PORT UPSTREAM_URL", file=sys.stderr)
        sys.exit(2)
    port = int(sys.argv[1])
    upstream = sys.argv[2].rstrip("/")
    server = ThreadedTCPServer(("127.0.0.1", port), make_handler(upstream))
    print(f"qwen36-merge-sys-proxy listening on 127.0.0.1:{port} -> {upstream}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
