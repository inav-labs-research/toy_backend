"""
Speech-to-speech inference handler.
"""
import threading
import numpy as np
from app.services.inferencing_handlers.inference_handler import InferenceHandler
from app.services.text_to_speech.base_tts_processor import BaseTTSProcessor
from app.media_stream_handler.icall_session_handler import ICallSessionHandler
from app.models.stt_models.base_realtime_stt import BaseRealtimeSTT
from app.models.language_models.online_serving.base_prediction_client import BasePredictionClient
from app.utils.logger import logger
from app.utilities.text_transformation import TextTransformation


class SpeechToSpeechHandler(InferenceHandler):
    """Speech-to-speech inference handler."""
    
    def __init__(
        self,
        llm_client: BasePredictionClient,
        tts_processor: BaseTTSProcessor,
        session_handler: ICallSessionHandler,
        streaming_stt_client: BaseRealtimeSTT = None,
        system_prompt: str = "",
        llm_config: dict = None,
    ):
        self.llm_client = llm_client
        self.tts_processor = tts_processor
        self.session_handler = session_handler
        self.streaming_stt_client = streaming_stt_client
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.llm_config = llm_config or {}

    async def process_text_for_inference(self, input_text: str, stop_event: threading.Event):
        """Process text for inference."""
        try:
            if not input_text or not input_text.strip():
                logger.warning("No text received for inference", "stt_interaction")
                return

            logger.info(f"Transcribed text to process: {input_text}", "stt_interaction")

            # Send user's transcribed text to frontend
            if hasattr(self.session_handler, 'send_json_message_async'):
                await self.session_handler.send_json_message_async("user_text", input_text)
            elif hasattr(self.session_handler, 'websocket'):
                import json
                user_text_message = json.dumps({"event_type": "user_text", "text": input_text})
                await self.session_handler.websocket.send_text(user_text_message)
            else:
                logger.info(f"User text (would send to frontend): {input_text}", "stt_interaction")

            # Add user message to history
            self.conversation_history.append({"role": "user", "content": input_text})

            # Prepare messages
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history[-10:])  # Keep last 10 messages

            # Prepare metadata for prediction client
            metadata = {
                "temperature": self.llm_config.get("temperature", 0.7),
                "max_tokens": self.llm_config.get("max_tokens", 1000)
            }

            # Stream LLM response and send to TTS in chunks (like pranthora)
            text_buffer_tts = []
            text_buffer_llm = []
            full_response = ""
            accumulated_text_for_display = ""
            
            async for response_chunk in self.llm_client.streaming_prediction(
                messages=messages,
                metadata=metadata
            ):
                if stop_event.is_set():
                    return
                
                if response_chunk.text:
                    output_text = response_chunk.text
                    text_buffer_tts.append(output_text)
                    text_buffer_llm.append(output_text)
                    full_response += output_text
                    accumulated_text_for_display += output_text
                    
                    # Send text chunk instantly to websocket for display
                    if accumulated_text_for_display.strip():
                        await self.session_handler.send_text_to_client_async(accumulated_text_for_display)
                    
                    # Check if we should send to TTS (20 words or sentence boundary)
                    current_text = "".join(text_buffer_tts)
                    words = current_text.strip().split()
                    word_count = len(words)
                    
                    # Send to TTS when we have enough words or hit sentence boundary
                    should_send = (
                        word_count >= 20 or 
                        any(current_text.rstrip().endswith(p) for p in ['.', '!', '?', '.\n', '!\n', '?\n'])
                    )
                    
                    if should_send and current_text.strip():
                        # Clean text before sending to TTS (remove reasoning tags, etc.)
                        cleaned_output = TextTransformation.clean_text_for_speech(current_text)
                        if cleaned_output:  # Only send if there's actual content after cleaning
                            logger.info(f"TTS output: {cleaned_output[:100]}", "tts_output")
                            await self._send_to_tts(cleaned_output, stop_event)
                        text_buffer_tts = []

            # Send any remaining text
            if text_buffer_tts:
                remaining_text = "".join(text_buffer_tts)
                cleaned_remaining = TextTransformation.clean_text_for_speech(remaining_text)
                if cleaned_remaining:
                    logger.info(f"TTS output: {cleaned_remaining[:100]}", "tts_output")
                    await self._send_to_tts(cleaned_remaining, stop_event)

            if full_response:
                # Add assistant response to history (keep original with reasoning for context)
                self.conversation_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"Error in voice interaction: {e}", "stt_interaction", exc_info=True)
            # Send error message to user
            try:
                await self._send_to_tts("Something went wrong while processing your request.", stop_event)
            except Exception as tts_error:
                logger.error(f"Error sending error message to TTS: {tts_error}", "stt_interaction")

    async def process_audio_for_inference(self, audio_data, stop_event: threading.Event):
        """Process audio for inference."""
        # For now, we use text-based inference with streaming STT
        # This can be extended to use audio directly
        if self.streaming_stt_client:
            transcript = self.streaming_stt_client.get_transcript()
            if transcript:
                await self.process_text_for_inference(transcript, stop_event)
        else:
            logger.warning("No STT client available for audio inference", "SpeechToSpeechHandler")

    async def _send_to_tts(self, text: str, stop_event: threading.Event):
        """Send text to TTS and stream audio to client."""
        try:
            # Check if text is only punctuation/symbols
            if not text or not text.strip() or all(c in '.,!?;: \n\t' for c in text.strip()):
                logger.debug(f"Skipping TTS for non-alphanumeric output: {text}", "tts_output")
                return
            
            async def audio_callback(audio_chunk: bytes):
                """Callback to send audio chunks to client."""
                if not stop_event.is_set():
                    await self.session_handler.send_audio_to_client_async(audio_chunk)

            await self.tts_processor.stream_text_to_speech(
                text=text,
                stop_event=stop_event,
                audio_async_callback=audio_callback
            )

        except Exception as e:
            logger.error(f"Error sending to tts: {e}", "stt_interaction", exc_info=True)

    async def cleanup(self):
        """Cleanup resources."""
        self.conversation_history = []
        logger.info("Cleaned up inference handler", "SpeechToSpeechHandler")

