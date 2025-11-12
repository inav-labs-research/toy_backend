"""
WebSocket stream handler for media streams.
"""
import asyncio
import json
import time
from typing import Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from app.services.handlers.base_realtime_voice_handler import BaseRealtimeVoiceHandler
from app.utils.static_memory_cache import StaticMemoryCache
from app.utils.logger import logger


class WebSocketStreamHandler:
    """Handler for WebSocket media streams."""
    
    def __init__(
        self,
        max_pending_chunks: int = 15,
        processing_delay_threshold: float = 0.05,
        cleanup_interval: float = 2.0,
        max_concurrent_tasks: int = 25
    ):
        self.max_pending_chunks = max_pending_chunks
        self.processing_delay_threshold = processing_delay_threshold
        self.cleanup_interval = cleanup_interval
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.active_tasks: Set[asyncio.Task] = set()
        self.should_stop = asyncio.Event()
        self.chunk_counter = 0
        self.last_cleanup = time.time()
        self.loop = asyncio.get_event_loop()
        self.session_timeout = StaticMemoryCache.get_config("general", "call_session_timeout")
        self.session_start_time = None

    async def handle_stream(
        self,
        websocket: WebSocket,
        real_time_handler: BaseRealtimeVoiceHandler,
    ):
        """Handle the WebSocket stream."""
        try:
            await websocket.accept()
            logger.info("Incoming web socket connection is established.", "web_socket_stream_handler")
            self.session_start_time = time.time()

            # Create timeout checker task
            timeout_checker = asyncio.create_task(self._check_session_timeout(websocket))

            # Send first response for web users immediately
            await real_time_handler.lazy_initialize()
            await real_time_handler.generate_first_response_from_agent("WEBSITE")

            await websocket.send_text(json.dumps({"event_type": "start_media_streaming"}))

            # Handle web audio stream
            while not self.should_stop.is_set():
                try:
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.5)
                    await self._process_incoming_data(data, real_time_handler)
                    await self._cleanup_tasks()
                except asyncio.TimeoutError:
                    # Check if we should stop (for graceful shutdown) - check more frequently
                    if self.should_stop.is_set():
                        break
                    continue
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected by client", "web_socket_stream_handler")
                    break
                except Exception as e:
                    logger.error(f"Error receiving data: {e}", "web_socket_stream_handler", exc_info=True)
                    break
                
        except WebSocketDisconnect:
            logger.error("Media stream disconnected.", "web_socket_stream_handler")
        except Exception as e:
            logger.error(f"Error in stream handling: {e}", "web_socket_stream_handler", exc_info=True)
            raise
        finally:
            # Signal stop to all components
            self.should_stop.set()
            
            # Cancel timeout checker
            if 'timeout_checker' in locals():
                timeout_checker.cancel()
                try:
                    await timeout_checker
                except asyncio.CancelledError:
                    pass
            
            # First: Cleanup voice handler (which closes Cartesia STT first)
            try:
                await real_time_handler.handle_voice_disconnect()
            except Exception as e:
                logger.error(f"Error in voice handler cleanup: {e}", "web_socket_stream_handler", exc_info=True)
            
            # Second: Close main websocket (after Cartesia is closed)
            try:
                await self.close_websocket(websocket, code=1000, reason="Connection closed")
            except Exception as e:
                logger.error(f"Error closing websocket: {e}", "web_socket_stream_handler", exc_info=True)
            
            # Finally: Cleanup tasks
            await self._cleanup()

    async def _process_incoming_data(self, data, real_time_handler):
        """Process incoming audio data."""
        self.chunk_counter += 1
        arrival_time = time.time()

        if len(self.active_tasks) > self.max_pending_chunks:
            logger.debug(f"Too many pending tasks ({len(self.active_tasks)}), skipping chunk {self.chunk_counter}", "web_socket_stream_handler")
            return

        task = asyncio.create_task(
            self._process_audio_chunk(data, self.chunk_counter, arrival_time, real_time_handler)
        )
        self.active_tasks.add(task)

    async def _process_audio_chunk(self, data: bytes, chunk_id: int, arrival_time: float, real_time_handler: BaseRealtimeVoiceHandler):
        """Process a single audio chunk."""
        try:
            elapsed = time.time() - arrival_time
            if elapsed > self.processing_delay_threshold:
                logger.debug(f"Skipping chunk {chunk_id} - too old ({elapsed:.3f}s)", "web_socket_stream_handler")
                return

            async with self.processing_semaphore:
                start_time = time.time()
                await real_time_handler.handle_web_audio_stream(data)
                processing_time = time.time() - start_time
                if processing_time > 0.02:
                    logger.debug(f"Chunk {chunk_id} processed in {processing_time:.3f}s", "web_socket_stream_handler")
        except Exception as e:
            logger.error(f"Error processing audio chunk {chunk_id}: {e}", "web_socket_stream_handler", exc_info=True)
        finally:
            self.active_tasks.discard(asyncio.current_task())

    async def _check_session_timeout(self, websocket: WebSocket):
        """Check for session timeout."""
        try:
            while not self.should_stop.is_set():
                await asyncio.sleep(1)
                if time.time() - self.session_start_time > self.session_timeout:
                    logger.info(f"Session timeout reached ({self.session_timeout}s). Closing connection.", "web_socket_stream_handler")
                    self.should_stop.set()
                    await self.close_websocket(websocket, code=1000, reason="Session timeout reached")
                    break
        except Exception as e:
            logger.error(f"Error in timeout checker: {e}", "web_socket_stream_handler", exc_info=True)
            self.should_stop.set()

    async def close_websocket(self, websocket: WebSocket, code: int = 1000, reason: str = "Normal Closure"):
        """Close the WebSocket connection."""
        try:
            # Check if websocket is still connected by checking client_state
            # WebSocketState values: CONNECTING, CONNECTED, DISCONNECTED
            from starlette.websockets import WebSocketState
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close(code=code, reason=reason)
                logger.info(f"WebSocket closed with code {code}: {reason}", "web_socket_stream_handler")
        except Exception as e:
            # If websocket is already closed or any other error, just log it
            logger.debug(f"WebSocket already closed or error closing: {e}", "web_socket_stream_handler")

    async def _cleanup_tasks(self):
        """Clean up completed tasks."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            done_tasks = {t for t in self.active_tasks if t.done()}
            self.active_tasks -= done_tasks
            logger.debug(f"Cleaned up {len(done_tasks)} completed tasks. Active: {len(self.active_tasks)}", "web_socket_stream_handler")
            self.last_cleanup = current_time

    async def _cleanup(self):
        """Clean up all tasks."""
        self.should_stop.set()
        for task in self.active_tasks:
            task.cancel()
        if self.active_tasks:
            try:
                await asyncio.wait(self.active_tasks, timeout=0.5)
            except Exception as e:
                logger.error(f"Error during cleanup: {e}", "web_socket_stream_handler", exc_info=True)

