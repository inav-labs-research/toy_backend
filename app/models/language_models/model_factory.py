"""
Model factory for creating prediction clients.
"""
from enum import Enum
from app.models.language_models.online_serving.clients.qwen_online_prediction_client import QwenPredictionClient
from app.models.language_models.online_serving.clients.llm_model_client import LLMModelClient


class OnlineModel(Enum):
    """Enum for online model types."""
    QWEN = "qwen"
    OPENAI = "openai"
    GEMINI = "gemini"


class ModelFactory:
    """Factory for creating prediction clients."""

    @staticmethod
    def create_online_model_client(
        online_model_type: OnlineModel,
        api_key: str,
        api_base: str,
        model_id: str,
        extra_args: dict = None,
        temperature: float = 0.6
    ):
        """
        Create an online model client based on the model type.
        
        Args:
            online_model_type: The type of online model (QWEN, OPENAI, GEMINI)
            api_key: API key for the model
            api_base: Base URL for the API
            model_id: Model ID/name
            extra_args: Extra arguments to pass to the model
            temperature: Temperature for the model
            
        Returns:
            BasePredictionClient: The appropriate prediction client
        """
        if online_model_type == OnlineModel.QWEN:
            # Ensure reasoning_efforts is set to "none" for Qwen
            if extra_args is None:
                extra_args = {}
            if "reasoning_efforts" not in extra_args:
                extra_args["reasoning_efforts"] = "none"
            return QwenPredictionClient(api_key, api_base, model_id, extra_args)
        elif online_model_type == OnlineModel.GEMINI:
            # Use LLMModelClient with gemini provider
            return LLMModelClient(
                api_key=api_key,
                api_base=api_base,
                model_name=model_id,
                temperature=temperature,
                extra_args=extra_args,
                model_provider="gemini"
            )
        elif online_model_type == OnlineModel.OPENAI:
            # Use LLMModelClient with openai provider
            return LLMModelClient(
                api_key=api_key,
                api_base=api_base,
                model_name=model_id,
                temperature=temperature,
                extra_args=extra_args,
                model_provider="openai"
            )
        else:
            raise ValueError(f'Model not found: {online_model_type}')

