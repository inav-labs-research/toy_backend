"""
Base class for realtime voice handlers.
"""
import threading
from abc import ABC, abstractmethod
from app.utils.logger import logger


class BaseRealtimeVoiceHandler(ABC):
    """Base class for realtime voice handlers."""

    @abstractmethod
    async def lazy_initialize(self):
        """Lazy initialization of resources."""
        pass

    @abstractmethod
    async def handle_user_audio_stream(self, raw_audio_base64: str):
        """Handle user audio stream (for Twilio)."""
        pass

    @abstractmethod
    async def handle_web_audio_stream(self, raw_audio_bytes: bytes):
        """Handle web audio stream."""
        pass

    @abstractmethod
    async def handle_voice_disconnect(self):
        """Handle voice disconnection."""
        pass

    async def generate_first_response_from_agent(self, user_input_source) -> None:
        """Generate and send the first agent response."""
        if hasattr(self, 'session_handler') and self.session_handler.first_response_message:
            first_message = self.session_handler.first_response_message
            logger.info(f"Generating first response for {user_input_source}: {first_message}", "first_response")
            try:
                # Send first response text to websocket immediately
                await self.session_handler.send_text_to_client_async(first_message)
                
                if hasattr(self, 'audio_infer_handler') and self.audio_infer_handler and hasattr(self.audio_infer_handler, '_send_to_tts'):
                    stop_event = threading.Event()
                    await self.audio_infer_handler._send_to_tts(first_message, stop_event)
                    logger.info("First response sent successfully", "first_response")
                else:
                    logger.warning("No inferencing handler available for first response", "first_response")
            except Exception as e:
                logger.error(f"Error generating first response: {str(e)}", "first_response", exc_info=True)

