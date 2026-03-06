import os
import json
import httpx
from typing import AsyncGenerator, Optional


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, model: Optional[str] = None):
        """Initialize Ollama client with configuration from environment or parameters."""
        self.host = host or os.getenv("OLLAMA_HOST", "ollama")
        self.port = port or int(os.getenv("OLLAMA_PORT", "11434"))
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        self.base_url = f"http://{self.host}:{self.port}"
    
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama."""
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream('POST', url, json=payload) as response:
                    response.raise_for_status()
                    
                    buffer = b""
                    async for raw_chunk in response.aiter_raw():
                        buffer += raw_chunk
                        
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            if line.strip():
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    if 'response' in data and data['response']:
                                        yield data['response']
                                    if data.get('done', False):
                                        return
                                except (json.JSONDecodeError, UnicodeDecodeError):
                                    continue
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")

