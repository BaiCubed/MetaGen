                                        
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import os
import time

                             
                           
                             

ChatMessage = Dict[str, str]                                                           

@runtime_checkable
class LLMClient(Protocol):
    def chat(self, messages: List[ChatMessage], temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, Any]:
        """
        Return:
          {
            "text": "<assistant text>",
            "usage": {
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int
            },
            # (optional) "raw": <provider-specific object>
          }
        """
        ...

@dataclass
class LLMConfig:
    provider: str                                                   
    model: str                                                                         
    api_key_env: str = "OPENAI_API_KEY"                                
    base_url: Optional[str] = None                                                      
    adapter_dir: Optional[str] = None                      
    dtype: str = "bfloat16"                                                                    
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_s: int = 60
    max_retries: int = 3
    backoff_s: float = 1.2


                             
                                          
                             

try:
    from openai import OpenAI
except Exception as _e:
    OpenAI = None                  
class _OpenAIChatClient(LLMClient):
    """
    OpenAI 兼容客户端（DeepSeek 也走这条），满足本项目统一的 LLMClient 接口：
      - __init__(cfg: LLMConfig)
      - chat(messages: List[Dict[str,str]], temperature: float = None, max_tokens: int = None) -> Dict

    读取配置优先级：
      1) cfg.api_key （若存在，直接使用，不推荐）
      2) os.environ[cfg.api_key_env] （推荐做法）
    """

    def __init__(self, cfg: LLMConfig):
        if OpenAI is None:
            raise ImportError("Missing dependency 'openai'. Please `pip install openai` first.")

                                                          
        api_key: Optional[str] = getattr(cfg, "api_key", None)
        if api_key and isinstance(api_key, str) and api_key.strip():
            self.api_key = api_key.strip()
        else:
            if not cfg.api_key_env:
                raise EnvironmentError("Missing 'api_key' or 'api_key_env' in LLM config.")
            env_val = os.environ.get(cfg.api_key_env)
            if not env_val:
                raise EnvironmentError(
                    f"Missing API key in env var '{cfg.api_key_env}'. Set it before running."
                )
            self.api_key = env_val.strip()

                      
        self.base_url: str = (cfg.base_url or "https://api.deepseek.com").rstrip("/")
        self.model: str = cfg.model
        self.default_temperature: float = float(getattr(cfg, "temperature", 0.2) or 0.2)
        self.default_max_tokens: int = int(getattr(cfg, "max_tokens", 512) or 512)
        self.timeout_s: float = float(getattr(cfg, "timeout_s", 60) or 60)
        self.max_retries: int = int(getattr(cfg, "max_retries", 3) or 3)
        self.backoff_s: float = float(getattr(cfg, "backoff_s", 1.2) or 1.2)

                                  
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

                              
    @staticmethod
    def _coerce_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        fixed: List[Dict[str, str]] = []
        for m in messages:
            role = str(m.get("role", "user"))
            content = m.get("content", "")
                                              
            if not isinstance(content, str):
                content = str(content)
            fixed.append({"role": role, "content": content})
        return fixed

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        执行一次非流式对话请求。
        返回：
          {
            "text": str,                          # 第一条候选的文本
            "usage": { "prompt_tokens": int,
                       "completion_tokens": int,
                       "total_tokens": int }
          }
        """
        msgs = self._coerce_messages(messages)
        temp = self._pick(temperature, self.default_temperature)
        mtok = self._pick(max_tokens, self.default_max_tokens)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                                                                     
                                 
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=float(temp),
                    max_tokens=int(mtok),
                    stream=False,
                                                                   
                    **({"timeout": self.timeout_s} if "timeout" in self.client.__dict__ else {}),
                )

                      
                text = ""
                if getattr(resp, "choices", None):
                    choice0 = resp.choices[0]
                                                    
                    msg = getattr(choice0, "message", None)
                    if msg is not None:
                        text = getattr(msg, "content", "") or ""
                    else:
                                                 
                        text = getattr(choice0, "text", "") or ""

                usage = getattr(resp, "usage", None)
                if usage is not None:
                    u = {
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    }
                else:
                                     
                    u = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                return {"text": text, "usage": u}

            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_s * attempt)

                       
        raise RuntimeError(f"OpenAI-compatible chat failed after {self.max_retries} retries: {last_err}")

                   
    @staticmethod
    def _pick(x: Optional[Any], default: Any) -> Any:
        return default if x is None else x

                             
                                 
                             

class _LocalPEFTLLMClient(LLMClient):
    """
    Wraps LocalPEFTClient (base + PEFT adapter) as an LLMClient.
    """
    def __init__(self, cfg: LLMConfig):
        if not cfg.adapter_dir:
            raise ValueError("For provider 'peft_local', llm.adapter_dir must be set to a trained adapter path.")
        try:
            from metagen_ai.utils.local_peft_client import LocalPEFTClient
        except Exception as e:
            raise RuntimeError("Missing LocalPEFTClient. Ensure 'metagen_ai/utils/local_peft_client.py' exists "
                               "and required packages (transformers, peft, torch) are installed.") from e
        self._client = LocalPEFTClient(base_model=cfg.model, adapter_dir=cfg.adapter_dir, dtype=cfg.dtype)
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens

    def chat(self, messages: List[ChatMessage], temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, Any]:
        temp = temperature if temperature is not None else self._temperature
        mxtok = max_tokens if max_tokens is not None else self._max_tokens
        out = self._client.chat(messages, temperature=float(temp), max_tokens=int(mxtok))
        return {"text": out["text"], "usage": out.get("usage", {}), "raw": None}


                             
         
                             

def _cfg_from_dict(cfg: Dict[str, Any]) -> LLMConfig:
    llm = cfg.get("llm", {})
    if not llm:
        raise ValueError("Missing 'llm' section in config.")
    return LLMConfig(
        provider=str(llm.get("provider", "")).strip(),
        model=str(llm.get("model", "")).strip(),
        api_key_env=str(llm.get("api_key_env", "OPENAI_API_KEY")),
        base_url=llm.get("base_url"),
        adapter_dir=llm.get("adapter_dir"),
        dtype=str(llm.get("dtype", "bfloat16")),
        temperature=float(llm.get("temperature", 0.2)),
        max_tokens=int(llm.get("max_tokens", 512)),
        timeout_s=int(llm.get("timeout_s", 60)),
        max_retries=int(llm.get("max_retries", 3)),
        backoff_s=float(llm.get("backoff_s", 1.2)),
    )

def build_llm_from_cfg(cfg: Dict[str, Any]) -> LLMClient:
    """
    Build an LLM client from a unified config dict (configs/default.yaml -> llm:*).
    Strict mode: raises on missing deps or invalid config; no silent fallbacks.
    """
    llmcfg = _cfg_from_dict(cfg)
    if not llmcfg.provider:
        raise ValueError("llm.provider must be specified: 'openai' | 'vllm' | 'peft_local'")
    if not llmcfg.model:
        raise ValueError("llm.model must be specified.")

    provider = llmcfg.provider.lower()
    if provider == "openai":
                                                     
        return _OpenAIChatClient(llmcfg)
    elif provider == "vllm":
                                                                     
        return _OpenAIChatClient(llmcfg)
    elif provider == "peft_local":
                                             
        return _LocalPEFTLLMClient(llmcfg)
    else:
        raise ValueError(f"Unknown llm.provider '{llmcfg.provider}'. Expected 'openai' | 'vllm' | 'peft_local'.")


                             
                         
                             

def render_system_user(system: str, user: str) -> List[ChatMessage]:
    """
    Convenience for simple 2-turn prompts.
    """
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
