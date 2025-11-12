"""
Base TTS processor interface.
"""
from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator
import threading


class BaseTTSProcessor(ABC):
    """Base class for TTS processors."""
    
    def __init__(self, language: str = 'en', slow: bool = False):
        self.language = language
        self.slow = slow
    
    @abstractmethod
    async def stream_text_to_speech(
        self,
        text: str,
        voice_name: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
        user_input_source=None,
        audio_callback: Optional[callable] = None,
        audio_async_callback: Optional[callable] = None
    ):
        """Stream text to speech."""
        pass
    
    @abstractmethod
    async def convert_text_to_speech_bytes(self, text: str, voice_name: Optional[str] = None, user_input_source=None) -> bytes:
        """Convert text to speech bytes."""
        pass
    
    @abstractmethod
    async def convert_text_to_speech_base64(self, text: str, voice_name: Optional[str] = None, user_input_source=None) -> str:
        """Convert text to speech base64."""
        pass
    
    @abstractmethod
    async def preconnect(self, voice_name: str = None, user_input_source=None):
        """Preconnect to TTS service."""
        pass
    
    @abstractmethod
    def close_connection(self):
        """Close TTS connection."""
        pass

