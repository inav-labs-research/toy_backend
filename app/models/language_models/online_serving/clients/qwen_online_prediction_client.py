"""
Qwen prediction client with reasoning_efforts support.
"""
import json
import time
from typing import Any, AsyncGenerator, Dict, List
from app.models.language_models.online_serving.base_prediction_client import BasePredictionClient
from app.models.language_models.online_serving.model_streaming_response import ModelStreamingResponse
from app.utils.logger import logger


class QwenPredictionClient(BasePredictionClient):
    """
    Implementation of the QwenPredictionClient that handles streaming predictions.
    """
    
    def __init__(self, api_key: str, api_base: str, model_id: str, extra_args: dict = None):
        super().__init__(api_key, api_base, model_id)
        self.event_name = "qwen_online_prediction"
        self.extra_args = extra_args or {}
        
        # Ensure reasoning_efforts is set to "none" for Qwen models
        if "reasoning_efforts" not in self.extra_args:
            self.extra_args["reasoning_efforts"] = "none"
        elif self.extra_args.get("reasoning_efforts") is None:
            self.extra_args["reasoning_efforts"] = "none"
   
    def get_tools(self):
        """Return a list of tools to be used in the prediction."""
        return []

    async def streaming_prediction(
        self, 
        messages: List[Dict[str, str]], 
        tools: str = None, 
        metadata: Dict[str, Any] = None
    ) -> AsyncGenerator[ModelStreamingResponse, None]:
        """Handle streaming prediction with Qwen model."""
        # Prepare kwargs following pranthora_backend pattern
        # Filter out invalid parameters before spreading
        filtered_extra_args = {k: v for k, v in self.extra_args.items() 
                              if k not in ["thinking", "platform", "reasoning_efforts"]}
        
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            **filtered_extra_args
        }
        
        tool_prompt = ""
        if metadata:
            kwargs["temperature"] = metadata.get("temperature", 0.7)
            if "max_tokens" in metadata:
                kwargs["max_tokens"] = metadata["max_tokens"]
            if "agent_tools" in metadata:
                kwargs["tools"] = metadata["agent_tools"]
            tool_prompt = metadata.get("agent_tool_prompt", "")
        
        if tools:
            kwargs["tools"] = tools
        
        # Handle reasoning_efforts separately - pass via extra_body if DeepInfra supports it
        # The OpenAI SDK doesn't recognize it, but DeepInfra might accept it in the request body
        reasoning_efforts = self.extra_args.get("reasoning_efforts", "none")
        if reasoning_efforts:
            # Try passing via extra_body for DeepInfra compatibility
            kwargs["extra_body"] = {"reasoning_efforts": reasoning_efforts}
        
        # Only log URL, not payload (for privacy and performance)
        logger.info(f"Qwen streaming prediction to: {self.api_base}/chat/completions", "qwen_online_prediction")
        
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
                        logger.info(f"Qwen first token time: {first_token_latency:.2f} ms", "qwen_online_prediction")
                        first_token_received = True
                    
                    yield ModelStreamingResponse(text=content)
        except GeneratorExit:
            # Handle generator cleanup gracefully
            raise
        except Exception as e:
            logger.error(f"Error in Qwen streaming prediction: {str(e)}", "qwen_online_prediction", exc_info=True)
            raise

    @staticmethod
    def _get_content_from_chunk(chunk: Any) -> str:
        """Extract content from a chat completion chunk."""
        delta = chunk.choices[0].delta
        return delta.content if hasattr(delta, "content") else None
    
    async def batch_prediction(self, messages: list, tools = []):
        """
        Handle batch prediction logic using OpenAI API's chat completions.
        This will return the response in a batch, not streaming.
        """
        # Filter out invalid parameters before spreading
        filtered_extra_args = {k: v for k, v in self.extra_args.items() 
                              if k not in ["thinking", "platform", "reasoning_efforts"]}
        
        kwargs = {
            "messages": messages,
            "model": self.model_id,
            "stream": False,
            **filtered_extra_args
        }
        
        if tools:
            kwargs["tools"] = tools
        
        # Handle reasoning_efforts separately - pass via extra_body if DeepInfra supports it
        reasoning_efforts = self.extra_args.get("reasoning_efforts", "none")
        if reasoning_efforts:
            kwargs["extra_body"] = {"reasoning_efforts": reasoning_efforts}
        
        chat_completion = await self.client.chat.completions.create(**kwargs)
        return chat_completion

