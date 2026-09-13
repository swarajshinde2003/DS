import json, os, uuid, time, logging
import requests
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_FILE = r"proxy_live.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("proxy")
def p(msg): log.info(msg); print(msg, flush=True)

# Replace with your Gemini API Key or set environment variable GEMINI_API_KEY
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
PROXY_PORT      = 8081

http_session = requests.Session()

MODEL_MAP = {
    "claude-opus-5":              "gemini-2.5-flash",
    "claude-sonnet-5":            "gemini-2.5-flash",
    "claude-haiku-5":             "gemini-2.5-flash",
    "claude-3-5-sonnet-20241022": "gemini-2.5-flash",
    "claude-3-5-haiku-20241022":  "gemini-2.5-flash",
    "sonnet":                     "gemini-2.5-flash",
    "haiku":                      "gemini-2.5-flash",
}
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

def map_model(name: str) -> str:
    return MODEL_MAP.get(name, DEFAULT_GEMINI_MODEL)

def content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                t = b.get("type", "")
                if t == "text":
                    parts.append(b.get("text", ""))
                elif t == "tool_use":
                    parts.append(f"[Tool: {b.get('name')}({json.dumps(b.get('input',{}))})]")
                elif t == "tool_result":
                    inner = b.get("content", "")
                    if isinstance(inner, list):
                        inner = " ".join(x.get("text","") for x in inner if isinstance(x,dict))
                    parts.append(f"[Result: {inner}]")
            else:
                parts.append(str(b))
        return "\n".join(p for p in parts if p)
    return str(content)

def anthropic_to_openai(data: dict):
    original_model = data.get("model", DEFAULT_GEMINI_MODEL)
    gmodel         = map_model(original_model)
    messages = []

    if "system" in data:
        sys_content = content_to_str(data["system"])
        if sys_content.strip():
            messages.append({"role": "system", "content": sys_content})

    for msg in data.get("messages", []):
        role = msg.get("role", "user")
        content = content_to_str(msg.get("content", ""))
        messages.append({"role": role, "content": content})

    out = {"model": gmodel, "messages": messages}
    if "max_tokens" in data: out["max_tokens"] = data["max_tokens"]
    if "temperature" in data: out["temperature"] = data["temperature"]
    if "stream" in data: out["stream"] = data["stream"]

    return out, gmodel

class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        p(f"[HTTP] {fmt % args}")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self):
        p(f"[PROXY GET] {self.path}")
        if self.path.startswith("/v1/models") or self.path.startswith("/models"):
            self._handle_models()
        else:
            self._send_mock_ok()

    def do_POST(self):
        p(f"[PROXY POST] {self.path}")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        if self.path.startswith("/v1/messages") or self.path.startswith("/messages"):
            self._handle_messages(body)
        else:
            self._send_mock_ok()

    def _send_mock_ok(self):
        resp_data = json.dumps({"status": "ok", "message": "Proxy healthy"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(resp_data)

    def _handle_models(self):
        anthropic_list = {
            "data": [
                {"type": "model", "id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet", "created_at": "2024-10-22T00:00:00Z"},
                {"type": "model", "id": "claude-sonnet-5", "display_name": "Claude Sonnet 5", "created_at": "2026-01-01T00:00:00Z"},
                {"type": "model", "id": "sonnet", "display_name": "Sonnet", "created_at": "2024-01-01T00:00:00Z"}
            ]
        }
        out = json.dumps(anthropic_list).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(out)

    def _handle_messages(self, body: bytes):
        t0 = time.time()
        try:
            anthropic_req = json.loads(body)
        except Exception as e:
            p(f"[ERR] Bad JSON: {e}")
            self.send_error(400, "Invalid JSON")
            return

        original_model = anthropic_req.get("model", DEFAULT_GEMINI_MODEL)
        is_stream      = anthropic_req.get("stream", False)

        openai_req, gemini_model = anthropic_to_openai(anthropic_req)
        p(f"[REQ] model={original_model} -> {gemini_model} stream={is_stream}")

        target  = f"{GEMINI_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type":  "application/json"
        }

        try:
            resp = http_session.post(target, json=openai_req, headers=headers, stream=is_stream, timeout=60)
        except Exception as e:
            p(f"[ERR] Gemini Connection Error: {e}")
            self.send_error(502, "Gemini API connection error")
            return

        if resp.status_code != 200:
            err_body = resp.content
            p(f"[ERR] Gemini HTTP {resp.status_code}: {err_body[:500]}")
            self.send_response(resp.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            try: self.wfile.write(err_body)
            except: pass
            return

        msg_id = "msg_" + str(uuid.uuid4()).replace("-", "")[:24]

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            # 1. message_start
            m_start = json.dumps({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": original_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1}
                }
            })
            self.wfile.write(f"event: message_start\ndata: {m_start}\n\n".encode("utf-8"))

            # 2. content_block_start
            cb_start = json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            })
            self.wfile.write(f"event: content_block_start\ndata: {cb_start}\n\n".encode("utf-8"))
            self.wfile.flush()

            # 3. Stream deltas
            out_tokens = 0
            for line_bytes in resp.iter_lines():
                if not line_bytes: continue
                lstr = line_bytes.decode("utf-8", errors="replace").strip()
                if not lstr or not lstr.startswith("data:"): continue
                raw_data = lstr[5:].strip()
                if raw_data == "[DONE]": break
                try:
                    oai_chunk = json.loads(raw_data)
                    choices = oai_chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            out_tokens += 1
                            chunk_data = json.dumps({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text}
                            })
                            self.wfile.write(f"event: content_block_delta\ndata: {chunk_data}\n\n".encode("utf-8"))
                            self.wfile.flush()
                except Exception: pass

            # 4. content_block_stop
            cb_stop = json.dumps({"type": "content_block_stop", "index": 0})
            self.wfile.write(f"event: content_block_stop\ndata: {cb_stop}\n\n".encode("utf-8"))

            # 5. message_delta
            m_delta = json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": out_tokens or 1}
            })
            self.wfile.write(f"event: message_delta\ndata: {m_delta}\n\n".encode("utf-8"))

            # 6. message_stop
            m_stop = json.dumps({"type": "message_stop"})
            self.wfile.write(f"event: message_stop\ndata: {m_stop}\n\n".encode("utf-8"))
            self.wfile.flush()
            p(f"[STREAM OK] Finished streaming in {time.time() - t0:.2f}s ({out_tokens} chunks)")
        else:
            try:
                oai_resp = resp.json()
                text = oai_resp["choices"][0]["message"]["content"]
            except Exception: text = "Hello!"

            anthropic_resp = {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": original_model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 10}
            }
            out = json.dumps(anthropic_resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(out)
            p(f"[NON-STREAM OK] Finished in {time.time() - t0:.2f}s")

def run_server():
    server = ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
    p(f"[INIT] Proxy listening on port {PROXY_PORT}...")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
