import os
import json
import httpx
from typing import AsyncGenerator, Optional, Dict


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, model: Optional[str] = None):
        """Initialize Ollama client with configuration from environment or parameters."""
        self.host = host or os.getenv("OLLAMA_HOST", "ollama")
        self.port = port or int(os.getenv("OLLAMA_PORT", "11434"))
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        self.base_url = f"http://{self.host}:{self.port}"
        self.system_prompt_config = self._load_system_prompt()
    
    def _load_system_prompt(self) -> Dict:
        """Load system prompt configuration from JSON file specified in environment."""
        prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "prompts/system_prompt.json")
        
        try:
            if not os.path.isabs(prompt_path):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                prompt_path = os.path.join(project_root, prompt_path)
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config
        except FileNotFoundError:
            raise RuntimeError(f"System prompt file not found: {prompt_path}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in system prompt file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading system prompt: {e}")
    
    def format_prompt(self, context: str, query: str) -> str:
        """Format the prompt template with context and query."""
        template = self.system_prompt_config.get("prompt_template", "")
        return template.format(context=context, query=query)
    
    def get_prompt_metadata(self) -> Dict:
        """Get metadata about the current system prompt."""
        return {
            "version": self.system_prompt_config.get("version", "unknown"),
            "metadata": self.system_prompt_config.get("metadata", {}),
            "config": self.system_prompt_config.get("config", {})
        }
    
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

