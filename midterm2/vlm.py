import base64, json, threading, time, os
import cv2
import numpy as np
import websocket  # pip install websocket-client
from dotenv import load_dotenv

load_dotenv()

REALTIME_URI = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"  # or gpt-realtime-mini

class RealtimeMazeSentry:
    """
    Sends occasional thumbnails to OpenAI Realtime and lets the model
    call a tool named `trigger_main` when it sees a maze.
    """
    def __init__(self, on_trigger, fps=1, cooldown_sec=10, model_uri=REALTIME_URI):
        self.on_trigger = on_trigger
        self.cooldown_sec = cooldown_sec
        self.last_trigger_t = 0
        self.ws = None
        self._stop = False
        self._send_every = max(1, int(30 // fps))  # if you call maybe_send each frame at ~30fps
        self._frame_i = 0
        self._open(model_uri)
        self.connected = False
        self.sent_count = 0
        self.target_fps = fps
        self._last_send = 0.0

    def _open(self, model_uri):
        headers = [
            "Authorization: Bearer " + os.environ.get("OPENAI_API_KEY", ""),
            "OpenAI-Beta: realtime=v1",
        ]
        self.ws = websocket.WebSocketApp(
            model_uri,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=lambda ws, e: print("Realtime error:", e),
            on_close=lambda ws, code, msg: print("Realtime closed:", code, msg),
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def _on_open(self, _ws):
        self.connected = True
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": (
                    "You are a visual sentry. You receive single image frames. "
                    "If a frame contains a maze, call the tool `trigger_main` with {\"confidence\": <0..1>}."
                ),
                "tools": [{
                    "type": "function",
                    "name": "trigger_main",
                    "description": "Invoke the app's main() when a maze is present.",
                    "parameters": {
                        "type": "object",
                        "properties": {"confidence": {"type": "number", "minimum": 0, "maximum": 1}},
                        "required": ["confidence"]
                    }
                }],
            },
        }
        _ws.send(json.dumps(session_update))
        print("Realtime session ready.")
    
    def _on_close(self, _ws, code, msg):
        self.connected = False
        print(f"[Realtime] closed code={code} msg={msg}")

    def _on_message(self, _ws, message):
        try:
            evt = json.loads(message)
        except Exception:
            return
        print("[RT evt]", evt.get("type"))  
        t = evt.get("type", "")

        if t == "error":
            # print the full object to see why it's unhappy
            print("[RT error]", json.dumps(evt, indent=2))
            return

        # Collect streaming function-call args
        if t == "response.function_call.arguments.delta":
            call_id = evt.get("call_id")
            name = evt.get("name", "")
            delta = evt.get("delta", "")
            if not call_id:
                return
            rec = self._fn_buf.setdefault(call_id, {"name": name, "args": ""})
            # name may arrive only once; keep the first non-empty
            if name and not rec["name"]:
                rec["name"] = name
            rec["args"] += delta
            return

        # Finalize when the call completes
        if t == "response.function_call.completed":
            call_id = evt.get("call_id")
            rec = self._fn_buf.pop(call_id, None)
            if not rec:
                return
            name = rec.get("name", "")
            if name != "trigger_main":
                return

            # Parse accumulated JSON args
            try:
                args = json.loads(rec.get("args", "") or "{}")
            except Exception:
                args = {}

            conf = float(args.get("confidence", 0.0))
            now = time.time()
            if conf >= 0.8 and (now - self.last_trigger_t) > self.cooldown_sec:
                self.last_trigger_t = now
                print(f"[Realtime] Maze detected; confidence={conf:.2f}. Calling main()…")
                threading.Thread(target=self.on_trigger, daemon=True).start()

        # (Optional) log when session is ready
        if t == "session.updated":
            print("[Realtime] session.updated")

    def stop(self):
        self._stop = True
        try:
            self.ws.close()
        except Exception:
            pass

    @staticmethod
    def _b64_jpeg(img_bgr, max_w=512):
        h, w = img_bgr.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    def maybe_send(self, frame_bgr):
        if not self.connected or self.ws is None:
            return

        now = time.time()
        interval = 1.0 / max(0.1, float(self.target_fps))
        if (now - self._last_send) < interval:
            return

        # Encode to base64 bytes (no data: URL)
        # ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        # if not ok:
        #     print("[Sentry] JPEG encode failed")
        #     return
        # b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            print("[Sentry] JPEG encode failed"); return
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        img_url = "data:image/jpeg;base64," + b64

        # 1) Put the message into the conversation
        item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Look only for printed/hand-drawn mazes (grid-like labyrinths). "
                            "If you see one in the image, call the tool `trigger_main` with "
                            "{\"confidence\": number between 0 and 1}. If no maze, do nothing."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": img_url,
                    },
                ],
            },
        }

        # 2) Ask the model to produce a response (text only, tools allowed)
        rsp = {
            "type": "response.create",
            "response": {
                "tool_choice": {"type": "auto"},    
                "modalities": ["text"],          
            },
        }

        try:
            self.ws.send(json.dumps(item))
            self.ws.send(json.dumps(rsp))
            self._last_send = now
            self.sent_count += 1
            if (self.sent_count % 5) == 0:
                print(f"[Sentry] sent frames: {self.sent_count}")
        except Exception as e:
            print("send error:", e)




