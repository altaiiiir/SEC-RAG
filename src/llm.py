import os
import json
import requests
from typing import AsyncGenerator, Optional


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        model: Optional[str] = None
    ):
        """Initialize Ollama client with configuration from environment or parameters."""
        self.host = host or os.getenv("OLLAMA_HOST", "ollama")
        self.port = port or int(os.getenv("OLLAMA_PORT", "11434"))
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:4b")
        self.base_url = f"http://{self.host}:{self.port}"
    
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from Ollama.
        
        Args:
            prompt: The prompt to send to the model
            
        Yields:
            Chunks of generated text
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        
        try:
            # Use requests with stream=True for streaming response
            response = requests.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            yield chunk['response']
                        
                        # Stop if done
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")
    
    def check_health(self) -> bool:
        """Check if Ollama service is healthy."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def pull_model(self, model: Optional[str] = None) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model: Model name to pull (defaults to configured model)
            
        Returns:
            True if successful, False otherwise
        """
        model_name = model or self.model
        url = f"{self.base_url}/api/pull"
        payload = {"name": model_name, "stream": False}
        
        try:
            response = requests.post(url, json=payload, timeout=600)
            return response.status_code == 200
        except:
            return False
