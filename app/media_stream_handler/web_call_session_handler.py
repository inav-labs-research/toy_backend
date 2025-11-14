"""
Web call session handler.
"""
import asyncio
from typing import Optional
from fastapi import WebSocket
from app.media_stream_handler.icall_session_handler import ICallSessionHandler
from app.media_stream_handler.websocket_stream_handler import WebSocketStreamHandler
from app.utils.logger import logger

VOICE_INTRUPTION_SIGNAL = "stop"


class WebCallSessionHandler(ICallSessionHandler):
    """Web client implementation."""

    def __init__(
        self,
        websocket: WebSocket,
        stream_handler: WebSocketStreamHandler,
        request_id: str,
        user_id: str,
        first_response_message: Optional[str] = None,
    ):
        self.websocket = websocket
        self.stream_handler = stream_handler
        self.request_id = request_id
        self.user_id = user_id
        self._first_response_message = first_response_message

    @property
    def first_response_message(self) -> Optional[str]:
        """Get the first response message for this session."""
        return self._first_response_message

    @first_response_message.setter
    def first_response_message(self, value: Optional[str]) -> None:
        """Set the first response message for this session."""
        self._first_response_message = value

    def interrupt_stream(self) -> None:
        """Interrupt detected (sync), notifying web client."""
        try:
            # Check if websocket is still connected
            from starlette.websockets import WebSocketState
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                logger.debug("WebSocket disconnected, skipping interrupt signal (sync)", "web_interrupt_sync")
                return
                
            logger.info("Interrupt detected (sync), notifying web client", "web_interrupt_sync")
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_text(VOICE_INTRUPTION_SIGNAL), asyncio.get_event_loop()
            )
        except RuntimeError as e:
            if "close message has been sent" in str(e):
                logger.debug("WebSocket already closed, skipping interrupt signal (sync)", "web_interrupt_sync")
            else:
                logger.error(f"Failed to send interrupt (sync): {e}", "web_interrupt_sync_error")
        except Exception as e:
            logger.error(f"Failed to send interrupt (sync): {e}", "web_interrupt_sync_error")

    def send_audio_to_client(self, audio_message) -> None:
        """Send audio to web client (sync)."""
        try:
            # Check if websocket is still connected
            from starlette.websockets import WebSocketState
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                logger.debug("WebSocket disconnected, skipping audio send (sync)", "web_send_audio_sync")
                return
                
            logger.debug("Sending audio to web client (sync)", "web_send_audio_sync")
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_bytes(getattr(audio_message, "message", audio_message)), self.stream_handler.loop
            )
        except RuntimeError as e:
            if "close message has been sent" in str(e):
                logger.debug("WebSocket already closed, skipping audio send (sync)", "web_send_audio_sync")
            else:
                logger.error(f"Failed to send audio (sync): {e}", "web_send_audio_sync_error")
        except Exception as e:
            logger.error(f"Failed to send audio (sync): {e}", "web_send_audio_sync_error")

    def terminate_call(self) -> None:
        """Terminate web call (sync)."""
        try:
            logger.info("Terminating web call (sync)", "web_call_terminate_sync")
            self.stream_handler.should_stop.set()
            asyncio.run_coroutine_threadsafe(
                self.websocket.close(code=1000, reason="Call terminated by agent"), asyncio.get_event_loop()
            )
        except Exception as e:
            logger.error(f"Failed to terminate call (sync): {e}", "web_terminate_sync_error")

    async def interrupt_stream_async(self) -> None:
        """Interrupt detected (async), notifying web client."""
        try:
            # Check if websocket is still connected
            from starlette.websockets import WebSocketState
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                logger.debug("WebSocket disconnected, skipping interrupt signal", "web_interrupt_async")
                return
                
            logger.info("Interrupt detected (async), notifying web client", "web_interrupt_async")
            await self.websocket.send_text(VOICE_INTRUPTION_SIGNAL)
        except RuntimeError as e:
            if "close message has been sent" in str(e):
                logger.debug("WebSocket already closed, skipping interrupt signal", "web_interrupt_async")
            else:
                logger.error(f"Failed to send interrupt (async): {e}", "web_interrupt_async_error")
        except Exception as e:
            logger.error(f"Failed to send interrupt (async): {e}", "web_interrupt_async_error")

    async def send_audio_to_client_async(self, audio_message) -> None:
        """Send audio to web client (async)."""
        try:
            # Check if websocket is still connected
            from starlette.websockets import WebSocketState
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                logger.debug("WebSocket disconnected, skipping audio send", "web_send_audio_async")
                return
                
            logger.debug("Sending audio to web client (async)", "web_send_audio_async")
            await self.websocket.send_bytes(getattr(audio_message, "message", audio_message))
        except RuntimeError as e:
            if "close message has been sent" in str(e):
                logger.debug("WebSocket already closed, skipping audio send", "web_send_audio_async")
            else:
                logger.error(f"Failed to send audio (async): {e}", "web_send_audio_async_error")
        except Exception as e:
            logger.error(f"Failed to send audio (async): {e}", "web_send_audio_async_error")

    async def terminate_call_async(self) -> None:
        """Terminate web call (async)."""
        try:
            logger.info("Terminating web call (async)", "web_call_terminate_async")
            self.stream_handler.should_stop.set()
            await self.websocket.close(code=1000, reason="Call terminated by agent")
        except Exception as e:
            logger.error(f"Failed to terminate call (async): {e}", "web_terminate_async_error")

    def send_text_to_client(self, text: str) -> None:
        """Send text message to web client (sync)."""
        try:
            # Check if websocket is still connected
            from starlette.websockets import WebSocketState
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                logger.debug("WebSocket disconnected, skipping text send (sync)", "web_send_text_sync")
                return
                
            import json
            message = json.dumps({"event_type": "llm_text", "text": text})
            logger.debug(f"Sending text to web client (sync): {text[:50]}...", "web_send_text_sync")
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_text(message), self.stream_handler.loop
            )
        except RuntimeError as e:
            if "close message has been sent" in str(e):
                logger.debug("WebSocket already closed, skipping text send (sync)", "web_send_text_sync")
            else:
                logger.error(f"Failed to send text (sync): {e}", "web_send_text_sync_error")
        except Exception as e:
            logger.error(f"Failed to send text (sync): {e}", "web_send_text_sync_error")

    async def send_text_to_client_async(self, text: str) -> None:
        """Send text message to web client (async)."""
        try:
            # Check if websocket is still connected
            from starlette.websockets import WebSocketState
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                logger.debug("WebSocket disconnected, skipping text send", "web_send_text_async")
                return
                
            import json
            message = json.dumps({"event_type": "llm_text", "text": text})
            logger.debug(f"Sending text to web client (async): {text[:50]}...", "web_send_text_async")
            await self.websocket.send_text(message)
        except RuntimeError as e:
            if "close message has been sent" in str(e):
                logger.debug("WebSocket already closed, skipping text send", "web_send_text_async")
            else:
                logger.error(f"Failed to send text (async): {e}", "web_send_text_async_error")
        except Exception as e:
            logger.error(f"Failed to send text (async): {e}", "web_send_text_async_error")

    async def send_json_message_async(self, event_type: str, text: str) -> None:
        """Send JSON message to web client (async)."""
        try:
            # Check if websocket is still connected
            from starlette.websockets import WebSocketState
            if self.websocket.client_state == WebSocketState.DISCONNECTED:
                logger.debug("WebSocket disconnected, skipping JSON send", "web_send_json_async")
                return
                
            import json
            message = json.dumps({"event_type": event_type, "text": text})
            logger.debug(f"Sending {event_type} to web client (async): {text[:50]}...", "web_send_json_async")
            await self.websocket.send_text(message)
        except RuntimeError as e:
            if "close message has been sent" in str(e):
                logger.debug("WebSocket already closed, skipping JSON send", "web_send_json_async")
            else:
                logger.error(f"Failed to send JSON message (async): {e}", "web_send_json_async_error")
        except Exception as e:
            logger.error(f"Failed to send JSON message (async): {e}", "web_send_json_async_error")

