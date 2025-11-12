"""
Realtime voice handler with interruption support.
"""
import asyncio
import base64
from datetime import datetime, timezone
import numpy as np
from app.models.stt_models.base_realtime_stt import BaseRealtimeSTT
from app.services.handlers.base_realtime_voice_handler import BaseRealtimeVoiceHandler
from app.services.inferencing_handlers.inference_handler import InferenceHandler
from app.services.speech_processor.eos_processor import EosProcessor
from app.media_stream_handler.icall_session_handler import ICallSessionHandler
from app.utils.logger import logger


class RealtimeVoiceHandler(BaseRealtimeVoiceHandler):
    """Realtime voice handler with interruption support."""
    
    event_name = "realtime_voice_handler"

    def __init__(
        self,
        session_handler: ICallSessionHandler,
        inferencing_handler: InferenceHandler,
        eos_handler: EosProcessor = None,
        audio_recording_enabled: bool = False,
        streaming_stt_client: BaseRealtimeSTT = None,
    ):
        self.audio_samples_for_processing = np.array([], dtype=np.float32)
        self.eos_processor = eos_handler if eos_handler else EosProcessor()
        self.is_user_started_conversation = False
        self.user_turn = False
        self.thread_event_map = {}
        self.session_handler = session_handler
        self.max_interruptions = 25  # Increased to make interruption less sensitive
        self.last_audio_received_at = datetime.now(timezone.utc)
        self.interruption_tolerance_duration = 1  # 1 second tolerance
        self.interruption_counter = 0
        self.model_sampling_rate = 16000
        self.input_web_sampling_rate = 16000
        self.truncate_audio_buffer_threshold = 16000 * 4  # 4 seconds
        self.last_interupption_time = None
        self.audio_samples_tailing = self.model_sampling_rate * 3  # 3 seconds
        self.audio_input_recording_enabled = audio_recording_enabled
        self.audio_infer_handler = inferencing_handler
        self.user_input_raw_audio = np.array([], dtype=np.float32)
        self.streaming_stt_client: BaseRealtimeSTT = streaming_stt_client
        self.last_transcript_time = None  # Track when we last processed a transcript
        self.transcript_grace_period = 0.5  # Ignore interruptions for 0.5s after processing transcript
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            # If no loop is running, we'll set it during lazy_initialize
            self.main_loop = None

    async def lazy_initialize(self):
        """Lazy initialization."""
        if self.main_loop is None:
            self.main_loop = asyncio.get_running_loop()
            
        if self.streaming_stt_client:
            # Set callback to process transcripts immediately when they arrive
            if hasattr(self.streaming_stt_client, 'set_on_final_transcript_callback'):
                self.streaming_stt_client.set_on_final_transcript_callback(
                    self._on_final_transcript,
                    self.main_loop
                )
            # Set callback for partial transcripts to detect interruptions early
            if hasattr(self.streaming_stt_client, 'set_on_partial_transcript_callback'):
                self.streaming_stt_client.set_on_partial_transcript_callback(
                    self._on_partial_transcript,
                    self.main_loop
                )
            await self.streaming_stt_client.initalize()
    
    async def _on_partial_transcript(self, partial_text: str):
        """Callback called immediately when STT partial transcript arrives (for early interruption)."""
        try:
            if not partial_text or not partial_text.strip():
                return
            
            # If agent is responding and user starts speaking (partial transcript), interrupt immediately
            # This happens BEFORE final transcript, allowing TTS to stop instantly
            if not self.user_turn and self.thread_event_map:
                logger.info(f"Early interruption detected from partial transcript: {partial_text}", self.event_name)
                await self.handle_interrupt_audio_stream()
                
        except Exception as e:
            logger.error(f"Error in _on_partial_transcript: {str(e)}", self.event_name, exc_info=True)
    
    async def _on_final_transcript(self, transcript: str):
        """Callback called immediately when STT final transcript arrives."""
        try:
            if not transcript or not transcript.strip():
                return
            
            logger.info(f"STT final transcript received immediately: {transcript}", self.event_name)
            
            # Check if this is an interruption (user speaking while agent is responding)
            # Only interrupt if:
            # 1. user_turn is False (agent was responding)
            # 2. We have active inference tasks (agent is actually responding)
            # 3. Enough time has passed since last transcript (not trailing audio from same utterance)
            is_interruption = False
            if not self.user_turn and self.thread_event_map:
                # Check grace period to avoid false interruptions from trailing audio
                if self.last_transcript_time:
                    time_since_last = (datetime.now(timezone.utc) - self.last_transcript_time).total_seconds()
                    # If transcript comes too soon after last one, it's likely trailing audio, not interruption
                    if time_since_last > self.transcript_grace_period:
                        is_interruption = True
                    else:
                        logger.debug(f"Ignoring potential interruption (trailing audio, {time_since_last:.2f}s since last)", self.event_name)
                else:
                    # No previous transcript, so this is likely an interruption
                    is_interruption = True
            
            if is_interruption:
                logger.info("User interrupting agent response", self.event_name)
                await self.handle_interrupt_audio_stream()
            else:
                # Normal turn - mark conversation started
                if not self.is_user_started_conversation and self.session_handler.first_response_message:
                    await self.session_handler.interrupt_stream_async()
                self.is_user_started_conversation = True
                # Set user_turn to True initially for normal turns
                self.user_turn = True
            
            # Process transcript immediately
            stop_event = asyncio.Event()
            
            task = asyncio.create_task(
                self.process_text_inference(
                    input_text=transcript,
                    stop_event=stop_event
                )
            )
            
            self.thread_event_map[task] = stop_event
            
            # Mark that agent is now responding (user_turn = False)
            # This allows future transcripts to be detected as interruptions
            self.user_turn = False
            self.last_transcript_time = datetime.now(timezone.utc)
            self.interruption_counter = 0  # Reset interruption counter
            
        except Exception as e:
            logger.error(f"Error in _on_final_transcript: {str(e)}", self.event_name, exc_info=True)

    async def handle_user_audio_stream(self, raw_audio_base64: str):
        """Handle user audio stream (for Twilio)."""
        # For web, we use handle_web_audio_stream
        pass

    async def handle_web_audio_stream(self, raw_audio_bytes: bytes):
        """Handle web audio stream."""
        if self.streaming_stt_client:
            self.streaming_stt_client.transcribe_stream(raw_audio_bytes)
        
        # Convert bytes to numpy array
        audio_samples = np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        await self.process_audio_stream(audio_samples)

    def capture_user_audio(self, audio_sample):
        """Capture user audio for recording."""
        if self.audio_input_recording_enabled:
            self.user_input_raw_audio = np.concatenate((self.user_input_raw_audio, audio_sample))
            if len(self.user_input_raw_audio) > 16000 * 5:
                self.user_input_raw_audio = np.array([], dtype=np.float32)

    async def process_audio_stream(self, audio_samples):
        """Process audio stream for VAD and end-of-speech detection."""
        self.capture_user_audio(audio_samples)
        self.audio_samples_for_processing = np.concatenate((self.audio_samples_for_processing, audio_samples))

        # If STT has callback, skip EOS/interruption logic - let STT handle everything
        has_stt_callback = (self.streaming_stt_client and 
                           hasattr(self.streaming_stt_client, 'set_on_final_transcript_callback'))
        
        if has_stt_callback:
            # Only handle first response interruption when user starts speaking
            (_, voice_detected_in_frame) = self.eos_processor.is_end_of_speech_detected(
                self.audio_samples_for_processing
            )
            
            if not self.is_user_started_conversation:
                if voice_detected_in_frame:
                    self.user_turn = True
                    self.is_user_started_conversation = True
                    if self.session_handler.first_response_message:
                        await self.session_handler.interrupt_stream_async()
            
            # For STT with callbacks, don't do EOS/interruption detection
            # The callback will handle transcript processing
            return

        # Original EOS-based logic for non-callback STT
        (is_eos_detected, voice_detected_in_frame) = self.eos_processor.is_end_of_speech_detected(
            self.audio_samples_for_processing
        )

        if not self.is_user_started_conversation:
            if voice_detected_in_frame:
                self.user_turn = True
                self.is_user_started_conversation = True
                if self.session_handler.first_response_message:
                    await self.session_handler.interrupt_stream_async()
            else:
                logger.debug("Skipping processing frames as user has not initiated speaking", self.event_name)
                return

        if not is_eos_detected:
            self.last_audio_received_at = datetime.now()

        if not await self.should_ignore_end_of_speech() and is_eos_detected and self.user_turn:
            logger.debug("End of speech detected", self.event_name)
            self.user_turn = False
            self.prepared_audio_buffer_for_model = np.copy(self.audio_samples_for_processing)
            self.truncate_audio_buffers()
            
            # Process with non-callback STT or audio inference
            if self.streaming_stt_client:
                stop_event = asyncio.Event()
                task = asyncio.create_task(
                    self.process_text_inference(
                        input_text=self.streaming_stt_client.get_transcript(), stop_event=stop_event
                    )
                )
                self.thread_event_map[task] = stop_event
            elif not self.streaming_stt_client:
                # No STT client, use audio inference
                stop_event = asyncio.Event()
                task = asyncio.create_task(
                    self.process_audio_inference(input_data=self.prepared_audio_buffer_for_model, stop_event=stop_event)
                )
                self.thread_event_map[task] = stop_event

        elif not is_eos_detected and not self.user_turn:
            # Check for user interruption
            should_interrupt = await self.should_interrupt_audio_stream()
            if should_interrupt:
                logger.debug("User Interruption is detected", self.event_name)
                await self.handle_interrupt_audio_stream()

    async def process_text_inference(self, input_text, stop_event):
        """Process text for inference."""
        try:
            if input_text and input_text.strip():
                await self.audio_infer_handler.process_text_for_inference(input_text=input_text, stop_event=stop_event)
        except Exception as e:
            logger.error(f"Error in text inference: {str(e)}", self.event_name, exc_info=True)

    async def process_audio_inference(self, input_data, stop_event):
        """Process audio for inference."""
        try:
            await self.audio_infer_handler.process_audio_for_inference(input_data, stop_event)
        except Exception as e:
            logger.error(f"Error in audio inference: {str(e)}", self.event_name, exc_info=True)

    async def should_interrupt_audio_stream(self):
        """Check if audio stream should be interrupted."""
        self.interruption_counter += 1
        return self.interruption_counter > self.max_interruptions

    async def should_ignore_end_of_speech(self):
        """Check if end of speech should be ignored."""
        if self.last_interupption_time:
            time_since_interupption = datetime.now() - self.last_interupption_time
            if time_since_interupption.total_seconds() < self.interruption_tolerance_duration:
                return True
        return False

    async def handle_interrupt_audio_stream(self):
        """Handle interrupted audio stream."""
        logger.info("Audio stream interrupted by user. Stopping all threads.", self.event_name)

        self.interruption_counter = 0
        for task, stop_event in self.thread_event_map.items():
            stop_event.set()
        await self.cleanup_tasks()
        self.last_interupption_time = datetime.now()
        self.truncate_audio_buffers()
        self.session_handler.interrupt_stream()

        if self.streaming_stt_client:
            self.streaming_stt_client.reset_transcript()

        self.user_turn = True

    async def cleanup_tasks(self):
        """Clean up tasks."""
        items_to_remove = []
        for item, stop_event in list(self.thread_event_map.items()):
            if isinstance(item, asyncio.Task):
                if item.done():
                    items_to_remove.append(item)
                else:
                    item.cancel()
                    items_to_remove.append(item)

        for item in items_to_remove:
            if item in self.thread_event_map:
                del self.thread_event_map[item]

    def truncate_audio_buffers(self):
        """Truncate audio buffers."""
        if len(self.audio_samples_for_processing) > self.truncate_audio_buffer_threshold:
            self.audio_samples_for_processing = self.audio_samples_for_processing[-self.audio_samples_tailing:]

    async def handle_voice_disconnect(self):
        """Handle voice disconnection - close Cartesia first, then cleanup."""
        logger.info("Handling voice disconnect", self.event_name)
        try:
            # First: Close Cartesia STT (most important - close external connection first)
            if self.streaming_stt_client:
                try:
                    await self.streaming_stt_client.cleanup()
                    logger.info("Cartesia STT cleaned up", self.event_name)
                except Exception as e:
                    logger.error(f"Error cleaning up STT client: {str(e)}", self.event_name, exc_info=True)
            
            # Second: Cancel all tasks
            for task, stop_event in self.thread_event_map.items():
                stop_event.set()
                if isinstance(task, asyncio.Task):
                    task.cancel()
            await self.cleanup_tasks()
            
        except Exception as e:
            logger.error(f"Error in voice disconnect: {str(e)}", self.event_name, exc_info=True)

