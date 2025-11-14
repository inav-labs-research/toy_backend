"""
Cartesia TTS processor implementation.
"""
import base64
import time
import threading
import asyncio
from cartesia import Cartesia
from typing import Optional, Dict, Any, AsyncGenerator

from app.services.text_to_speech.base_tts_processor import BaseTTSProcessor
from app.utils.logger import logger


class CartesiaTTSConfig:
    """Configuration for Cartesia TTS."""
    def __init__(self, **config):
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Cartesia API key is required in config")
        self.model_id = config.get("model_name", "sonic-3")
        self.voice_id = config.get("default_voice", "cc00e582-ed66-4004-8336-0175b85c85f6")
        self.api_base = config.get("api_base", "https://api.cartesia.ai")
        self.api_version = config.get("api_version", "2025-04-16")
        
        # Generation config for speed, volume, and emotion
        self.speed = config.get("speed", 1.0)  # Default 0.7, range 0.6-1.5
        self.volume = config.get("volume", 1.0)  # Default 1.0, range 0.5-2.0
        self.emotion = config.get("emotion", None)  # Optional emotion guidance
        
        # For web clients
        self.web_encoding = config.get("web_encoding", "pcm_s16le")
        self.web_sample_rate = config.get("web_sample_rate", 16000)
        
        # For Twilio (phone calls)
        self.twilio_encoding = config.get("twilio_encoding", "pcm_mulaw")
        self.twilio_sample_rate = config.get("twilio_sample_rate", 8000)
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation config for TTS with speed, volume, and emotion."""
        gen_config = {}
        if self.speed is not None:
            # Clamp speed to valid range (0.6 to 1.5)
            gen_config["speed"] = max(0.6, min(1.5, self.speed))
        if self.volume is not None:
            # Clamp volume to valid range (0.5 to 2.0)
            gen_config["volume"] = max(0.5, min(2.0, self.volume))
        if self.emotion is not None:
            gen_config["emotion"] = self.emotion
        return gen_config if gen_config else None

    def get_output_format(self, user_input_source=None) -> Dict[str, Any]:
        """Get output format based on input source."""
        # For now, always use web format
        return {
            "container": "raw",
            "encoding": self.web_encoding,
            "sample_rate": self.web_sample_rate,
        }


class CartesiaTTSProcessor(BaseTTSProcessor):
    """Cartesia TTS processor."""
    
    def __init__(self, config: dict = None, language: str = 'en', slow: bool = False, voice_name: Optional[str] = None):
        super().__init__(language, slow)
        self.config = CartesiaTTSConfig(**config)
        self.tts_type = "cartesia"
        self.voice_id = voice_name if voice_name and str(voice_name).strip() else self.config.voice_id
        logger.info(f"Initialized CartesiaTTSProcessor with voice_id: {self.voice_id}", "cartesia_tts_init")

    def _sync_stream_tts(
        self,
        text: str,
        voice_id: Optional[str],
        user_input_source=None,
        stop_event: Optional[threading.Event] = None
    ):
        """Synchronous TTS streaming (to be run in executor)."""
        resolved_voice_id = voice_id if voice_id and str(voice_id).strip() else self.voice_id
        logger.info(f"Streaming TTS from Cartesia SDK with voice_id: {resolved_voice_id}", "cartesia_tts_stream")

        def _to_bytes(data) -> bytes:
            if isinstance(data, (bytes, bytearray, memoryview)):
                return bytes(data)
            if isinstance(data, str):
                try:
                    return base64.b64decode(data, validate=True)
                except Exception:
                    return data.encode("latin1", errors="ignore")
            return b""

        try:
            client = Cartesia(api_key=self.config.api_key)
            
            # Prepare generation config with speed, volume, and emotion
            generation_config = self.config.get_generation_config()
            
            # Build TTS request parameters
            tts_params = {
                "model_id": self.config.model_id,
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": resolved_voice_id,
                },
                "language": self.language,
                "output_format": self.config.get_output_format(user_input_source),
            }
            
            # Add generation_config if we have speed/volume/emotion settings
            if generation_config:
                tts_params["generation_config"] = generation_config
            
            response = client.tts.sse(**tts_params)

            chunks = []
            try:
                for chunk in response:
                    if stop_event and stop_event.is_set():
                        logger.info("Cartesia TTS streaming stopped by event.", "cartesia_tts_stream")
                        break
                    data = getattr(chunk, "data", chunk)
                    data_bytes = _to_bytes(data)
                    if data_bytes:
                        chunks.append(data_bytes)
            except GeneratorExit:
                # Handle generator cleanup gracefully
                logger.info("Cartesia TTS generator closed", "cartesia_tts_stream")
                raise
            except Exception as e:
                logger.error(f"Error in TTS generator: {str(e)}", "cartesia_tts_stream", exc_info=True)
                raise
            
            return chunks

        except Exception as e:
            logger.error(f"cartesia_tts_stream_request_failed: {str(e)}", "cartesia_tts_stream_request_failed", exc_info=True)
            raise
    
    async def _stream_tts_and_yield_chunks(
        self,
        text: str,
        voice_id: Optional[str],
        user_input_source=None,
        stop_event: Optional[threading.Event] = None
    ) -> AsyncGenerator[bytes, None]:
        """Async wrapper for TTS streaming."""
        # Run sync TTS in executor to avoid blocking
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None, 
            self._sync_stream_tts,
            text,
            voice_id,
            user_input_source,
            stop_event
        )
        
        # Yield chunks asynchronously
        for chunk in chunks:
            yield chunk

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
        start_time = time.time()
        try:
            async for audio_chunk in self._stream_tts_and_yield_chunks(text, voice_name, user_input_source, stop_event):
                # Call only one callback to avoid duplicate playback
                if audio_async_callback is not None:
                    await audio_async_callback(audio_chunk)
                elif audio_callback is not None:
                    audio_callback(audio_chunk)
        except Exception as e:
            logger.error(f"Streaming error: {e}", "cartesia_tts_stream_error", exc_info=True)
            raise
        finally:
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"cartesia_tts_done: Completed in {elapsed:.2f} ms", "cartesia_tts_done")

    async def convert_text_to_speech_bytes(self, text: str, voice_name: Optional[str] = None, user_input_source=None) -> bytes:
        """Convert text to speech bytes."""
        audio_chunks = []
        async for chunk in self._stream_tts_and_yield_chunks(text, voice_name, user_input_source):
            audio_chunks.append(chunk)
        return b"".join(audio_chunks)

    async def convert_text_to_speech_base64(self, text: str, voice_name: Optional[str] = None, user_input_source=None) -> str:
        """Convert text to speech base64."""
        audio_bytes = await self.convert_text_to_speech_bytes(text, voice_name, user_input_source)
        return base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""

    async def preconnect(self, voice_name: str = None, user_input_source=None):
        """Preconnect - no persistent connection required."""
        logger.info("No persistent connection required.", "cartesia_tts_preconnect")

    def close_connection(self):
        """Close connection - stateless API."""
        logger.info("Stateless API — nothing to close.", "cartesia_tts_close")

