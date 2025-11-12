"""
Base inference handler interface.
"""
from abc import ABC, abstractmethod
import threading


class InferenceHandler(ABC):
    """Base class for inference handlers."""

    @abstractmethod
    async def process_text_for_inference(self, input_text: str, stop_event: threading.Event):
        """Process text for inference."""
        pass

    @abstractmethod
    async def process_audio_for_inference(self, audio_data, stop_event: threading.Event):
        """Process audio for inference."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources."""
        pass

