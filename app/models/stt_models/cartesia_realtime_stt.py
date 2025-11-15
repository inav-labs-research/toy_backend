"""
Cartesia realtime STT implementation.
Optimized real-time STT wrapper using Cartesia Async API with buffering.
"""
import asyncio
import numpy as np
import time
from typing import Optional, Union
from cartesia import AsyncCartesia
from app.models.stt_models.base_realtime_stt import BaseRealtimeSTT
from app.utils.logger import logger


class CartesiaRealtimeSTT(BaseRealtimeSTT):
    """Optimized real-time STT wrapper using Cartesia Async API with buffering."""
    
    def __init__(
        self, 
        model_name: str = "ink-whisper",
        api_key: str = "",
        language: str = "en",
        sample_rate: int = 16000,
    ) -> None:
        self.api_key = api_key
        self.model = model_name
        self.language = language
        self.sample_rate = sample_rate
        self.encoding = "pcm_s16le"
        
        self.client: Optional[AsyncCartesia] = None
        self.ws = None
        self.is_connected = False

        # Transcript buffers
        self._accumulated_transcript = ""
        self._current_partial = ""

        self.last_result_time = None
        self._receive_task: Optional[asyncio.Task] = None
                
        # Audio buffering to reduce send frequency
        self.audio_buffer = b""
        self.buffer_duration_ms = 100  # Buffer 100ms of audio before sending
        self.bytes_per_ms = (self.sample_rate * 2) // 1000  # 2 bytes per sample (16-bit PCM)
        
        # Background task for sending buffered audio
        self._audio_send_task: Optional[asyncio.Task] = None
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        # Callback for when final transcript is ready
        self._on_final_transcript_callback = None
        # Callback for partial transcripts (for early interruption detection)
        self._on_partial_transcript_callback = None
        self._main_loop = None
        
    def set_on_final_transcript_callback(self, callback, loop=None):
        """Set callback to be called immediately when final transcript arrives."""
        self._on_final_transcript_callback = callback
        self._main_loop = loop
    
    def set_on_partial_transcript_callback(self, callback, loop=None):
        """Set callback to be called when partial transcript arrives (for early interruption)."""
        self._on_partial_transcript_callback = callback
        if not self._main_loop:
            self._main_loop = loop
        
    async def initalize(self):
        """Connect to Cartesia STT websocket."""
        self.client = AsyncCartesia(api_key=self.api_key)
        
        logger.info("Connecting to Cartesia...", "CartesiaRealtimeSTT")
        self.ws = await self.client.stt.websocket(
            model=self.model,
            language=self.language,
            encoding=self.encoding,
            sample_rate=self.sample_rate,
        )
        
        self.is_connected = True
        logger.info("Connected to Cartesia STT", "CartesiaRealtimeSTT")
        
        # Start background receiving
        self._receive_task = asyncio.create_task(self._receive_results())
        
        # Start background audio sender
        self._audio_send_task = asyncio.create_task(self._audio_sender())

    async def _audio_sender(self):
        """Background task that sends buffered audio to Cartesia."""
        try:
            while self.is_connected or not self._audio_queue.empty():
                try:
                    # Get audio chunk from queue with timeout
                    audio_chunk = await asyncio.wait_for(
                        self._audio_queue.get(), 
                        timeout=0.2
                    )
                    
                    if self.ws and self.is_connected:
                        try:
                            await self.ws.send(audio_chunk)
                            logger.debug(f"Sent audio: {len(audio_chunk)} bytes", "CartesiaRealtimeSTT")
                        except Exception as send_error:
                            logger.error(f"Error sending audio: {send_error}", "CartesiaRealtimeSTT")
                            self.is_connected = False
                            break
                            
                except asyncio.TimeoutError:
                    # No audio in queue, continue waiting
                    continue
                        
        except asyncio.CancelledError:
            logger.debug("Audio sender task cancelled", "CartesiaRealtimeSTT")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error in audio sender: {e}", "CartesiaRealtimeSTT")
            self.is_connected = False

    def transcribe_stream(self, audio_data: Union[bytes, np.ndarray]):
        """
        Non-blocking method to queue audio data for transcription.
        This method is NOT async - it just queues the data.
        """
        if not self.is_connected or not self.ws:
            logger.debug("Not connected, skipping audio", "CartesiaRealtimeSTT")
            return

        # Convert numpy array to bytes if needed
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            audio_data = audio_data.tobytes()

        try:
            # Add to buffer
            self.audio_buffer += audio_data
            target_buffer_size = self.buffer_duration_ms * self.bytes_per_ms
            
            # When buffer reaches target size, queue it for sending
            while len(self.audio_buffer) >= target_buffer_size:
                chunk = self.audio_buffer[:target_buffer_size]
                self.audio_buffer = self.audio_buffer[target_buffer_size:]
                
                # Non-blocking queue put
                try:
                    self._audio_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    logger.warning("Audio queue full, dropping chunk", "CartesiaRealtimeSTT")
            
        except Exception as e:
            logger.error(f"Error queueing audio: {e}", "CartesiaRealtimeSTT")
    
    async def _receive_results(self):
        """Receive partial and final transcripts asynchronously."""
        try:
            async for result in self.ws.receive():
                self.last_result_time = time.time()

                if result["type"] == "transcript":
                    text = result.get("text", "").strip()
                    if not text:
                        continue

                    if result.get("is_final"):
                        # Accumulate finalized speech
                        if self._accumulated_transcript:
                            self._accumulated_transcript += " " + text
                        else:
                            self._accumulated_transcript = text
                        self._current_partial = ""
                        logger.info(f"✓ Final segment: {text}", "CartesiaRealtimeSTT")
                        logger.debug(f"Accumulated: {self._accumulated_transcript}", "CartesiaRealtimeSTT")
                        
                        # Immediately call callback if set (process transcript without waiting for EOS)
                        if self._on_final_transcript_callback:
                            full_transcript = self._accumulated_transcript
                            # Clear accumulated so get_transcript() doesn't return it again
                            self._accumulated_transcript = ""
                            
                            # Call callback in main event loop
                            try:
                                if self._main_loop and self._main_loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        self._on_final_transcript_callback(full_transcript),
                                        self._main_loop
                                    )
                                else:
                                    # If no loop specified, try to schedule in current loop
                                    asyncio.create_task(self._on_final_transcript_callback(full_transcript))
                            except Exception as e:
                                logger.error(f"Error calling transcript callback: {e}", "CartesiaRealtimeSTT")
                    else:
                        # Track partial interim
                        self._current_partial = text
                        logger.debug(f"Partial: {text}", "CartesiaRealtimeSTT")
                        
                        # Call partial callback immediately for early interruption detection
                        # This allows interruption to happen BEFORE final transcript arrives
                        if self._on_partial_transcript_callback:
                            try:
                                if self._main_loop and self._main_loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        self._on_partial_transcript_callback(text),
                                        self._main_loop
                                    )
                                else:
                                    # If no loop specified, try to schedule in current loop
                                    asyncio.create_task(self._on_partial_transcript_callback(text))
                            except Exception as e:
                                logger.error(f"Error calling partial transcript callback: {e}", "CartesiaRealtimeSTT")

                elif result["type"] == "done":
                    logger.info("Transcription completed", "CartesiaRealtimeSTT")
                    break

        except Exception as e:
            logger.error(f"Error receiving results: {e}", "CartesiaRealtimeSTT")
        finally:
            self.is_connected = False

    def get_transcript(self) -> str:
        """
        Get and clear the accumulated transcription.
        Returns all final transcripts accumulated since last call, then clears the accumulation.
        For interruption detection, also includes current partial transcript if no final exists yet.
        """
        # If we have accumulated transcripts, return and clear them
        if self._accumulated_transcript:
            transcript = self._accumulated_transcript
            self._accumulated_transcript = ""
            logger.debug(f"Returning accumulated transcript: '{transcript[:100]}...'", "CartesiaRealtimeSTT")
            return transcript
        
        # Otherwise return current partial transcript (for interruption detection)
        # but don't clear it as it's not final yet
        if self._current_partial:
            logger.debug(f"Returning partial transcript: '{self._current_partial[:100]}...'", "CartesiaRealtimeSTT")
            return self._current_partial
        
        return ""
    
    def reset_transcript(self):
        """Reset all transcriptions. Called during interruption to clear state."""
        old_accumulated = self._accumulated_transcript
        old_current = self._current_partial
        self._accumulated_transcript = ""
        self._current_partial = ""
        logger.debug(f"Transcript reset (was: accumulated='{old_accumulated[:50] if old_accumulated else 'empty'}...', current='{old_current[:50] if old_current else 'empty'}...')", "CartesiaRealtimeSTT")
            
    async def cleanup(self):
        """Close websocket and clean up Cartesia client."""
        try:
            self.is_connected = False
            
            # Cancel audio sender task
            if self._audio_send_task and not self._audio_send_task.done():
                self._audio_send_task.cancel()
                try:
                    await asyncio.wait_for(self._audio_send_task, timeout=0.2)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Cancel receive task
            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await asyncio.wait_for(self._receive_task, timeout=0.2)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Close websocket
            if self.ws:
                await self.ws.close()
                
            # Close client
            if self.client:
                await self.client.close()

            # Clean up references
            self.ws = None
            self.client = None
            self._accumulated_transcript = ""
            self._current_partial = ""
            self.audio_buffer = b""
            self._receive_task = None
            self._audio_send_task = None
            
            logger.info("Cleaned up Cartesia connection", "CartesiaRealtimeSTT")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}", "CartesiaRealtimeSTT")
        finally:
            self.is_connected = False

