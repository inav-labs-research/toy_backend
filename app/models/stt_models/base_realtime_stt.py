"""
Base class for realtime STT implementations.
"""
from abc import ABC, abstractmethod


class BaseRealtimeSTT(ABC):
    """Base class for realtime speech-to-text implementations."""
    
    @abstractmethod
    def initalize(self):
        """Initialize and start the streaming session."""
        pass

    @abstractmethod
    def transcribe_stream(self, audio_data: bytes):
        """Send audio data for transcription."""
        pass

    @abstractmethod
    def get_transcript(self) -> str:
        """Get the current transcription."""
        pass

    @abstractmethod
    def reset_transcript(self):
        """Reset the current transcription."""
        pass

    @abstractmethod
    def cleanup(self):
        """Stop and cleanup the streaming session."""
        pass

