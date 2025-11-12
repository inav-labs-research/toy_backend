"""
Interface for call session handlers.
"""
from abc import ABC, abstractmethod
from typing import Optional


class ICallSessionHandler(ABC):
    """Interface for handling call session operations."""

    @property
    @abstractmethod
    def first_response_message(self) -> Optional[str]:
        """Get the first response message for this session."""
        pass

    @first_response_message.setter
    @abstractmethod
    def first_response_message(self, value: Optional[str]) -> None:
        """Set the first response message for this session."""
        pass

    @abstractmethod
    def interrupt_stream(self) -> None:
        """Interrupt the current audio stream (sync)."""
        pass

    @abstractmethod
    def send_audio_to_client(self, audio_message) -> None:
        """Send audio to the client (sync)."""
        pass

    @abstractmethod
    def terminate_call(self) -> None:
        """Terminate the current call (sync)."""
        pass

    @abstractmethod
    async def interrupt_stream_async(self) -> None:
        """Interrupt the current audio stream (async)."""
        pass

    @abstractmethod
    async def send_audio_to_client_async(self, audio_message) -> None:
        """Send audio to the client (async)."""
        pass

    @abstractmethod
    async def terminate_call_async(self) -> None:
        """Terminate the current call (async)."""
        pass

    @abstractmethod
    def send_text_to_client(self, text: str) -> None:
        """Send text message to the client (sync)."""
        pass

    @abstractmethod
    async def send_text_to_client_async(self, text: str) -> None:
        """Send text message to the client (async)."""
        pass

