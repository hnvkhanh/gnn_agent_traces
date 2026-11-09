import time

import requests

HOST = "http://localhost:11434"


def list_models(host: str = HOST):
    r = requests.get(host + "/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


class OllamaChat:
    """Callable: messages -> assistant text (JSON mode)."""

    def __init__(self, model: str, host: str = HOST, temperature: float = 0.7, num_predict: int = 200, timeout: int = 120):
        self.model, self.host, self.timeout = model, host, timeout
        self.options = {"temperature": temperature, "num_predict": num_predict}

    def __call__(self, messages):
        body = {"model": self.model, "messages": messages, "stream": False, "format": "json", "options": self.options}
        last = None
        for attempt in range(5):  # transient 500s happen under parallel load; back off and retry
            try:
                r = requests.post(self.host + "/api/chat", json=body, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["message"]["content"]
            except Exception as e:
                last = e
                time.sleep(min(8, 2 ** attempt))
        raise RuntimeError("ollama call failed: %r" % last)
