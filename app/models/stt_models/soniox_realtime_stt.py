"""
Soniox realtime STT implementation.
Async Soniox real-time STT with buffered streaming, modeled after CartesiaRealtimeSTT.
"""
import asyncio
import json
import numpy as np
from typing import Optional, Union
from websockets.client import connect as ws_connect
from app.data_layer.data_classes.domain_models.user_input_source import UserInputSource
from app.models.stt_models.base_realtime_stt import BaseRealtimeSTT
from app.utils.logger import logger


class SonioxRealtimeSTT(BaseRealtimeSTT):
    """Async Soniox real-time STT with buffered streaming, modeled after CartesiaRealtimeSTT."""
    
    def __init__(
        self, 
        api_key: str,
        model_name: str = "stt-rt-preview-v2",
        user_input_source: UserInputSource = UserInputSource.WEBSITE,
        language_hints: Optional[list] = None,
        sample_rate: int = 16000,
        enable_language_identification: bool = True,
        enable_speaker_diarization: bool = False,
        enable_endpoint_detection: bool = True,
        context: str = "",
        translation: Optional[dict] = None,
        num_channels: int = 1,
        audio_format: str = "pcm_s16le",
    ):
        self.api_key = api_key
        self.model = model_name
        self.user_input_source = user_input_source
        # Both WEBSITE and DEVICE use high-quality audio settings
        self.sample_rate = sample_rate
        self.encoding = audio_format
        self.language_hints = language_hints or ["hi"]
        self.enable_language_identification = enable_language_identification
        self.enable_speaker_diarization = enable_speaker_diarization
        self.enable_endpoint_detection = enable_endpoint_detection
        self.context = context
        self.translation = translation or {}
        
        self.websocket_url = "wss://stt-rt.soniox.com/transcribe-websocket"
        self.ws = None
        self.is_connected = False

        # Transcript buffers
        self._accumulated_transcript = ""
        self._current_partial = ""

        # Async background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._audio_send_task: Optional[asyncio.Task] = None

        # Audio queue and buffer
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.audio_buffer = b""
        self.buffer_duration_ms = 100
        self.bytes_per_ms = (self.sample_rate * 2) // 1000  # 16-bit PCM

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
        """Connect and start Soniox WebSocket session."""
        config = self._get_config()

        logger.info("Connecting to Soniox STT...", "SonioxRealtimeSTT")
        self.ws = await ws_connect(self.websocket_url)
        self.is_connected = True

        await self.ws.send(json.dumps(config))
        logger.info("Connected and configuration sent", "SonioxRealtimeSTT")

        # Start async background tasks
        self._receive_task = asyncio.create_task(self._receive_results())
        self._audio_send_task = asyncio.create_task(self._audio_sender())

    async def _audio_sender(self):
        """Send buffered audio chunks to Soniox."""
        try:
            while self.is_connected or not self._audio_queue.empty():
                try:
                    chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                    await self.ws.send(chunk)
                    logger.debug(f"Sent {len(chunk)} bytes of audio", "SonioxRealtimeSTT")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error sending audio: {e}", "SonioxRealtimeSTT")
                    self.is_connected = False
                    break

            # Send empty message to mark end of stream
            try:
                await self.ws.send("")
            except Exception:
                pass
        except asyncio.CancelledError:
            logger.debug("Audio sender cancelled", "SonioxRealtimeSTT")

    async def _receive_results(self):
        """Receive and process Soniox transcription messages."""
        try:
            async for message in self.ws:
                res = json.loads(message)
                
                # Error handling
                if res.get("error_code"):
                    msg = f"Soniox error {res['error_code']}: {res.get('error_message')}"
                    logger.error(msg, "SonioxRealtimeSTT")
                    break
                
                final_tokens, non_final_tokens = [], []
                for token in res.get("tokens", []):
                    if token.get("is_final"):
                        final_tokens.append(token)
                    else:
                        non_final_tokens.append(token)
                
                # Handle partials
                if non_final_tokens:
                    partial_text = self._render_tokens([], non_final_tokens)
                    if partial_text.strip():
                        self._current_partial = partial_text
                        logger.debug(f"Partial: {partial_text}", "SonioxRealtimeSTT")
                        
                        # Call partial callback immediately for early interruption detection
                        if self._on_partial_transcript_callback:
                            try:
                                if self._main_loop and self._main_loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        self._on_partial_transcript_callback(partial_text),
                                        self._main_loop
                                    )
                                else:
                                    # If no loop specified, try to schedule in current loop
                                    asyncio.create_task(self._on_partial_transcript_callback(partial_text))
                            except Exception as e:
                                logger.error(f"Error calling partial transcript callback: {e}", "SonioxRealtimeSTT")
                
                # Handle finals
                if final_tokens:
                    final_text = self._render_tokens(final_tokens, [])
                    if final_text.strip():
                        self._accumulated_transcript += (" " + final_text).strip()
                        self._current_partial = ""
                        logger.info(f"✓ Final: {final_text}", "SonioxRealtimeSTT")
                        
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
                                logger.error(f"Error calling transcript callback: {e}", "SonioxRealtimeSTT")

                if res.get("finished"):
                    logger.info("Session finished", "SonioxRealtimeSTT")
                    break
                
        except asyncio.CancelledError:
            logger.debug("Receive task cancelled", "SonioxRealtimeSTT")
        except Exception as e:
            logger.error(f"Receive loop error: {e}", "SonioxRealtimeSTT")
        finally:
            self.is_connected = False
    
    def transcribe_stream(self, audio_data: Union[bytes, np.ndarray]):
        """Queue audio for sending asynchronously."""
        if not self.is_connected:
            logger.debug("Not connected, skipping audio", "SonioxRealtimeSTT")
            return

        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            audio_data = audio_data.tobytes()

        self.audio_buffer += audio_data
        target_buffer_size = self.buffer_duration_ms * self.bytes_per_ms

        while len(self.audio_buffer) >= target_buffer_size:
            chunk = self.audio_buffer[:target_buffer_size]
            self.audio_buffer = self.audio_buffer[target_buffer_size:]
            try:
                self._audio_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping chunk", "SonioxRealtimeSTT")

    def get_transcript(self):
        """Return accumulated + current partial transcript."""
        if self._accumulated_transcript:
            t = self._accumulated_transcript
            self._accumulated_transcript = ""
            return t
        return self._current_partial or ""

    def reset_transcript(self):
        """Clear transcript buffers."""
        self._accumulated_transcript = ""
        self._current_partial = ""
        logger.debug("Transcript reset", "SonioxRealtimeSTT")

    async def cleanup(self):
        """Gracefully close websocket and cancel tasks."""
        try:
            self.is_connected = False
            for task in [self._receive_task, self._audio_send_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=2)
                    except asyncio.TimeoutError:
                        pass

            if self.ws:
                await self.ws.close()

            self.ws = None
            logger.info("Soniox connection cleaned up", "SonioxRealtimeSTT")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}", "SonioxRealtimeSTT")
    
    def _get_config(self):
        """Prepare Soniox websocket config message."""
        config = {
            "api_key": self.api_key,
            "model": self.model,
            "language_hints": self.language_hints,
            "enable_language_identification": self.enable_language_identification,
            "enable_speaker_diarization": self.enable_speaker_diarization,
            "enable_endpoint_detection": self.enable_endpoint_detection,
            "audio_format": self.encoding,
            "sample_rate": self.sample_rate,
            "num_channels": 1,
        }
        if self.context:
            config["context"] = self.context
        if self.translation:
            config["translation"] = self.translation
        return config
    
    def _render_tokens(self, final_tokens, non_final_tokens):
        """Render Soniox tokens into readable text."""
        ignore_tokens = {"<end>", "<start>", "<s>", "</s>"}
        text_parts = []
        
        for t in final_tokens + non_final_tokens:
            txt = t.get("text", "")
            if txt and txt.strip() not in ignore_tokens:
                text_parts.append(txt)
        
        return "".join(text_parts).strip()

