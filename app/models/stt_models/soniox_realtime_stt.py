"""
Soniox realtime STT implementation.
"""
import json
import threading
import queue
import numpy as np
from typing import Optional
from websockets.sync.client import connect
from websockets import ConnectionClosedOK
from app.models.stt_models.base_realtime_stt import BaseRealtimeSTT
from app.utils.logger import logger


class SonioxRealtimeSTT(BaseRealtimeSTT):
    """Soniox STT implementation with real-time streaming support."""
    
    def __init__(
        self, 
        model_name: str = "stt-rt-preview-v2",
        api_key: str = "",
        sample_rate: int = 16000,
        num_channels: int = 1,
        audio_format: str = "pcm_s16le",
        language_hints: Optional[list] = None,
        enable_language_identification: bool = True,
        enable_speaker_diarization: bool = False,
        enable_endpoint_detection: bool = True,
        context: str = "",
        translation: Optional[dict] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model_name
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.audio_format = audio_format
        self.language_hints = language_hints if language_hints else ["en"]
        self.enable_language_identification = enable_language_identification
        self.enable_speaker_diarization = enable_speaker_diarization
        self.enable_endpoint_detection = enable_endpoint_detection
        self.context = context
        self.translation = translation or {}
        
        self.websocket_url = "wss://stt-rt.soniox.com/transcribe-websocket"
        self.client = None
        self.is_connected = False
        self.audio_queue = queue.Queue()
        self.current_transcription = ""
        self.final_transcription = ""
        self.streaming_thread = None
        self._transcript_lock = threading.Lock()
        
    def initalize(self):
        """Initialize and start the Soniox streaming session."""
        try:
            if self.is_connected:
                logger.warning("Already connected to Soniox", "SonioxRealtimeSTT")
                return
                
            # Start streaming in a separate thread
            self.streaming_thread = threading.Thread(target=self._run_streaming_session, daemon=True)
            self.streaming_thread.start()
            
            # Wait a bit for connection to establish
            import time
            time.sleep(0.5)
            
            logger.info("Started Soniox streaming session", "SonioxRealtimeSTT")
            
        except Exception as e:
            logger.error(f"Error initializing streaming: {str(e)}", "SonioxRealtimeSTT", exc_info=True)
            raise
    
    def transcribe_stream(self, audio_data: bytes):
        """Send audio data to Soniox for transcription."""
        try:
            if not self.is_connected:
                logger.warning("Not connected to Soniox, dropping audio data", "SonioxRealtimeSTT")
                return
                
            # Handle numpy arrays if needed
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype != np.int16:
                    audio_data = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_data.tobytes()
            else:
                audio_bytes = audio_data
            
            self.audio_queue.put(audio_bytes)
            
        except Exception as e:
            logger.error(f"Error sending audio: {str(e)}", "SonioxRealtimeSTT", exc_info=True)
    
    def get_transcript(self) -> str:
        """Get the current transcription."""
        with self._transcript_lock:
            return self.final_transcription if self.final_transcription else self.current_transcription
        
    def reset_transcript(self):
        """Reset the current transcription."""
        with self._transcript_lock:
            self.current_transcription = ""
            self.final_transcription = ""
        logger.debug("Transcript reset", "SonioxRealtimeSTT")
        
    def cleanup(self):
        """Stop Soniox streaming session."""
        try:
            self.is_connected = False
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
            logger.info("Stopped Soniox streaming session", "SonioxRealtimeSTT")
        except Exception as e:
            logger.error(f"Error stopping streaming: {str(e)}", "SonioxRealtimeSTT", exc_info=True)
    
    def _run_streaming_session(self):
        """Run the streaming session in a separate thread."""
        try:
            config = self._get_config()
            
            logger.info("Connecting to Soniox...", "SonioxRealtimeSTT")
            with connect(self.websocket_url) as ws:
                self.client = ws
                self.is_connected = True
                
                # Send first request with config
                ws.send(json.dumps(config))
                
                # Start audio streaming thread
                audio_thread = threading.Thread(target=self._stream_audio, args=(ws,), daemon=True)
                audio_thread.start()
                
                logger.info("Session started", "SonioxRealtimeSTT")
                
                try:
                    final_tokens = []
                    while self.is_connected:
                        try:
                            message = ws.recv()
                        except Exception as e:
                            if not self.is_connected:
                                break
                            logger.debug(f"Error receiving message: {str(e)}", "SonioxRealtimeSTT")
                            continue
                            
                        res = json.loads(message)
                        
                        # Handle error from server
                        if res.get("error_code") is not None:
                            error_msg = f"Error: {res['error_code']} - {res.get('error_message', 'Unknown error')}"
                            logger.error(error_msg, "SonioxRealtimeSTT")
                            break
                        
                        # Parse tokens from current response
                        non_final_tokens = []
                        for token in res.get("tokens", []):
                            if token.get("text"):
                                if token.get("is_final"):
                                    final_tokens.append(token)
                                else:
                                    non_final_tokens.append(token)
                        
                        # Handle partial transcripts
                        if non_final_tokens:
                            partial_text = self._render_tokens([], non_final_tokens)
                            if partial_text.strip():
                                with self._transcript_lock:
                                    self.current_transcription = partial_text
                                logger.debug(f"Partial transcript: {partial_text}", "SonioxRealtimeSTT")
                        
                        # Handle final transcripts
                        if final_tokens:
                            final_text = self._render_tokens(final_tokens, [])
                            if final_text.strip():
                                with self._transcript_lock:
                                    self.final_transcription = final_text
                                    self.current_transcription = ""
                                logger.info(f"Final transcript: {final_text}", "SonioxRealtimeSTT")
                            final_tokens = []
                        
                        # Check for end of turn
                        if res.get("finished"):
                            logger.info("Session finished", "SonioxRealtimeSTT")
                            if final_tokens:
                                final_text = self._render_tokens(final_tokens, [])
                                with self._transcript_lock:
                                    self.final_transcription = final_text
                            break
                
                except ConnectionClosedOK:
                    logger.info("Connection closed normally", "SonioxRealtimeSTT")
                except Exception as e:
                    logger.error(f"Error in streaming session: {str(e)}", "SonioxRealtimeSTT", exc_info=True)
                finally:
                    self.is_connected = False
                    
        except Exception as e:
            logger.error(f"Error running streaming session: {str(e)}", "SonioxRealtimeSTT", exc_info=True)
            self.is_connected = False
    
    def _stream_audio(self, ws):
        """Stream audio data to the websocket."""
        try:
            while self.is_connected:
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    ws.send(audio_data)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error streaming audio: {str(e)}", "SonioxRealtimeSTT", exc_info=True)
                    break
            
            # Send empty string to signal end of audio
            try:
                ws.send("")
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error in audio streaming: {str(e)}", "SonioxRealtimeSTT", exc_info=True)
    
    def _get_config(self) -> dict:
        """Get Soniox configuration."""
        config = {
            "api_key": self.api_key,
            "model": self.model,
            "language_hints": self.language_hints,
            "enable_language_identification": self.enable_language_identification,
            "enable_speaker_diarization": self.enable_speaker_diarization,
            "enable_endpoint_detection": self.enable_endpoint_detection,
        }
        
        if self.context:
            config["context"] = self.context
        
        if self.translation:
            config["translation"] = self.translation
        
        if self.audio_format == "pcm_s16le":
            config["audio_format"] = "pcm_s16le"
            config["sample_rate"] = self.sample_rate
            config["num_channels"] = self.num_channels
        else:
            config["audio_format"] = self.audio_format
        
        return config
    
    def _render_tokens(self, final_tokens: list, non_final_tokens: list) -> str:
        """Convert tokens into a readable transcript."""
        text_parts = []
        current_speaker = None
        current_language = None
        
        for token in final_tokens + non_final_tokens:
            text = token.get("text", "")
            if not text:
                continue
                
            speaker = token.get("speaker")
            language = token.get("language")
            is_translation = token.get("translation_status") == "translation"
            
            if speaker is not None and speaker != current_speaker:
                if current_speaker is not None:
                    text_parts.append("\n\n")
                current_speaker = speaker
                current_language = None
                text_parts.append(f"Speaker {current_speaker}:")
            
            if language is not None and language != current_language:
                current_language = language
                prefix = "[Translation] " if is_translation else ""
                text_parts.append(f"\n{prefix}[{current_language}] ")
                text = text.lstrip()
            
            text_parts.append(text)
        
        return "".join(text_parts)

