"""
Base prediction client for LLM models.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from openai import AsyncOpenAI

from app.models.language_models.online_serving.model_streaming_response import ModelStreamingResponse


class BasePredictionClient(ABC):
    """
    Abstract base class for prediction clients (streaming and batch).
    Handles initialization and provides methods for making streaming and batch predictions.
    """
    def __init__(self, api_key: str, api_base: str, model_id: str):
        """
        Initialize the prediction client with API key, base URL, and model ID.
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model_id = model_id
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        self.tools = self.get_tools()

    @abstractmethod
    def get_tools(self):
        """Abstract method to return a list of tools to be used in the prediction."""
        pass

    @abstractmethod
    async def streaming_prediction(
        self, 
        messages: list, 
        tools: str = None, 
        metadata: dict = None
    ) -> AsyncGenerator[ModelStreamingResponse, None]:
        """
        Abstract method for streaming prediction. This will be implemented by subclasses.
        It should handle the prediction process where results are streamed.
        
        Args:
            messages (list): List of message objects for the conversation
            tools (str, optional): Tools to be used in the prediction
            metadata (dict, optional): Additional metadata for the prediction
            
        Returns:
            AsyncGenerator[ModelStreamingResponse]: Generator that yields ModelStreamingResponse objects
        """
        pass

    @abstractmethod
    async def batch_prediction(self, messages: list, tools = []):
        """
        Abstract method for batch prediction. This will be implemented by subclasses.
        It should handle the prediction process where results are returned all at once.
        """
        pass

