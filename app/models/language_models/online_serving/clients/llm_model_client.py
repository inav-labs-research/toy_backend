"""
LLM Model Client wrapper that supports multiple providers.
"""
import time
from typing import Any, AsyncGenerator, Dict, List
from app.models.language_models.online_serving.base_prediction_client import BasePredictionClient
from app.models.language_models.online_serving.model_streaming_response import ModelStreamingResponse
from app.models.language_models.online_serving.clients.gemini_prediction_client import GeminiPredictionClient
from app.utils.logger import logger


class LLMModelClient(BasePredictionClient):
    """
    Wrapper client that supports multiple LLM providers (OpenAI-compatible, Qwen, Gemini, etc.).
    """
    
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model_name: str,
        temperature: float = 0.6,
        extra_args: dict = None,
        model_provider: str = "openai"
    ):
        super().__init__(api_key, api_base, model_name)
        self.event_name = "llm_model_client"
        self.extra_args = extra_args or {}
        self.temperature = temperature
        self.model_provider = model_provider.lower()
        
        # Initialize provider-specific client
        if self.model_provider == "gemini":
            self.gemini_client = GeminiPredictionClient({
                "api_key": api_key,
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": extra_args.get("max_tokens", 1000) if extra_args else 1000,
                "api_base": api_base
            })
        else:
            self.gemini_client = None
   
    def get_tools(self):
        """Return a list of tools to be used in the prediction."""
        return []

    async def streaming_prediction(
        self, 
        messages: List[Dict[str, str]], 
        tools: str = None, 
        metadata: Dict[str, Any] = None
    ) -> AsyncGenerator[ModelStreamingResponse, None]:
        """Handle streaming prediction with tool support."""
        # Use Gemini client if provider is Gemini
        if self.model_provider == "gemini" and self.gemini_client:
            async for text in self.gemini_client.generate_response_streaming(messages):
                yield ModelStreamingResponse(text=text)
            return
        
        # Use OpenAI-compatible client for other providers (including Qwen)
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "temperature": metadata.get("temperature", self.temperature) if metadata else self.temperature,
            **self.extra_args
        }
        
        if metadata:
            if "max_tokens" in metadata:
                kwargs["max_tokens"] = metadata["max_tokens"]
            if "agent_tools" in metadata:
                kwargs["tools"] = metadata["agent_tools"]
        
        if tools:
            kwargs["tools"] = tools
        
        # Record start time for latency measurements
        start_time = time.time()
        token_count = 0
        first_token_received = False
        
        try:
            chat_completion_stream_response = await self.client.chat.completions.create(**kwargs)
            
            async for chunk in chat_completion_stream_response:
                content = self._get_content_from_chunk(chunk)
                if content:
                    token_count += 1
                    # Measure first token latency
                    if not first_token_received:
                        first_token_latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                        logger.info(f"LLM first token time: {first_token_latency:.2f} ms", self.event_name)
                        first_token_received = True
                    
                    yield ModelStreamingResponse(text=content)
        except Exception as e:
            logger.error(f"Error in streaming prediction: {str(e)}", self.event_name, exc_info=True)
            raise

    @staticmethod
    def _get_content_from_chunk(chunk: Any) -> str:
        """Extract content from a chat completion chunk."""
        delta = chunk.choices[0].delta
        return delta.content if hasattr(delta, "content") else None
    
    async def batch_prediction(self, messages: list, tools = [], should_return_content_only: bool = False):
        """
        Handle batch prediction logic using OpenAI API's chat completions.
        This will return the response in a batch, not streaming.
        """
        # Use Gemini client if provider is Gemini
        if self.model_provider == "gemini" and self.gemini_client:
            response = await self.gemini_client.generate_response(messages)
            return response if not should_return_content_only else response
        
        kwargs = {
            "messages": messages,
            "model": self.model_id,
            "temperature": self.temperature,
            "stream": False,
            **self.extra_args
        }
        
        if tools:
            kwargs["tools"] = tools
        
        chat_completion = await self.client.chat.completions.create(**kwargs)

        if should_return_content_only:
            return chat_completion.choices[0].message.content
        return chat_completion

