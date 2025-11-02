import time, threading
import cv2
import torch
import numpy as np
import open_clip
from PIL import Image

class OpenClipSentry:
    def __init__(self, on_trigger, threshold=0.80, fps=1.0, cooldown_sec=10,
                 model_name="ViT-B-32", pretrained="openai", device=None, prompts=None):
        """
        on_trigger: callback to run (e.g., lambda: main(hub, md))
        threshold: probability that "maze" is present (0..1)
        fps: how often to evaluate frames
        cooldown_sec: debounce between triggers
        """
        self.on_trigger = on_trigger
        self.threshold = float(threshold)
        self.cooldown = float(cooldown_sec)
        self._last_fire = 0.0
        self._last_send = 0.0
        self.interval = 1.0 / max(0.1, float(fps))

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()

        # Simple prompt ensemble improves robustness
        pos_prompts = [
            "a photo of a printed maze",
            "a black-and-white labyrinth drawing",
            "a page with a maze puzzle",
            "an image of a maze on paper"
        ]
        neg_prompts = [
            "not a maze", "a blank sheet of paper", "random shapes", "a photo without a maze"
        ]
        if prompts:  # allow custom overrides
            pos_prompts, neg_prompts = prompts

        tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            text = tokenizer(pos_prompts + neg_prompts).to(self.device)
            tfeat = self.model.encode_text(text)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        self.t_pos = tfeat[:len(pos_prompts)].mean(dim=0, keepdim=True)  # (1, D)
        self.t_neg = tfeat[len(pos_prompts):].mean(dim=0, keepdim=True)  # (1, D)
        self.text_feats = torch.cat([self.t_pos, self.t_neg], dim=0)     # (2, D)
        print(f"[OpenCLIP] ready on {self.device} (threshold={self.threshold}, fps={fps})")

    def _score(self, frame_bgr):
        # BGR -> RGB PIL
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        image = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device == "cuda":
                # mixed precision helps on GPU
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    ifeat = self.model.encode_image(image)
            else:
                ifeat = self.model.encode_image(image)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)  # (1, D)
            logits = (100.0 * ifeat @ self.text_feats.T).softmax(dim=-1)  # (1,2)
            p_maze = float(logits[0, 0].item())
        return p_maze

    def maybe_send(self, frame_bgr):
        now = time.time()
        if (now - self._last_send) < self.interval:
            return
        self._last_send = now

        p = self._score(frame_bgr)
        # lightweight console pulse
        print(f"[OpenCLIP] p(maze)={p:.2f}")
        if p >= self.threshold and (now - self._last_fire) > self.cooldown:
            self._last_fire = now
            print("[OpenCLIP] Maze detected. Triggering callback…")
            threading.Thread(target=self.on_trigger, daemon=True).start()