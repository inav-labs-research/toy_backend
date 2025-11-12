"""
Factory for creating agent handlers.
"""
from app.agents.agent_loader import AgentLoader
from app.models.stt_models.cartesia_realtime_stt import CartesiaRealtimeSTT
from app.models.stt_models.soniox_realtime_stt import SonioxRealtimeSTT
from app.services.text_to_speech.cartesia_tts_processor import CartesiaTTSProcessor
from app.services.inferencing_handlers.speech_to_speech_handler import SpeechToSpeechHandler
from app.services.handlers.realtime_voice_handler import RealtimeVoiceHandler
from app.services.speech_processor.eos_processor import EosProcessor
from app.media_stream_handler.icall_session_handler import ICallSessionHandler
from app.models.language_models.model_factory import ModelFactory, OnlineModel
from app.models.stt_models.base_realtime_stt import BaseRealtimeSTT
from app.utils.static_memory_cache import StaticMemoryCache
from app.utils.logger import logger


class AgentHandlerFactory:
    """Factory for creating agent handlers."""

    @staticmethod
    async def create_voice_handler_for_agent(
        agent_id: str,
        session_handler: ICallSessionHandler,
    ) -> RealtimeVoiceHandler:
        """Create voice handler for an agent."""
        try:
            # Load agent config
            agent_config = AgentLoader.get_agent(agent_id)
            if not agent_config:
                raise ValueError(f"Agent {agent_id} not found")

            # Get config
            config = StaticMemoryCache.config

            # Create LLM prediction client using model factory
            llm_config = config.get("models", {}).get("llm_model", {})
            model_provider = llm_config.get("model_provider", "openai").lower()
            platform = llm_config.get("platform", "").lower()
            
            # Determine model type
            if model_provider == "qwen" or platform == "deepinfra":
                model_type = OnlineModel.QWEN
            elif model_provider == "gemini":
                model_type = OnlineModel.GEMINI
            else:
                model_type = OnlineModel.OPENAI
            
            # Prepare extra_args
            extra_args = {}
            if llm_config.get("thinking") is not None:
                extra_args["thinking"] = llm_config.get("thinking")
            if llm_config.get("reasoning_efforts"):
                extra_args["reasoning_efforts"] = llm_config.get("reasoning_efforts")
            if llm_config.get("platform"):
                extra_args["platform"] = llm_config.get("platform")
            
            # Create prediction client
            llm_client = ModelFactory.create_online_model_client(
                online_model_type=model_type,
                api_key=llm_config.get("api_key", ""),
                api_base=llm_config.get("api_base", "https://api.openai.com/v1"),
                model_id=llm_config.get("model_id") or llm_config.get("model_name", "gpt-4o-mini"),
                extra_args=extra_args,
                temperature=llm_config.get("temperature", 0.7)
            )

            # Create TTS processor
            tts_config = config.get("models", {}).get("tts_model", {})
            tts_language = tts_config.get("language", "en")
            tts_processor = CartesiaTTSProcessor(config=tts_config, language=tts_language)

            # Create STT client - check which provider is active
            models_config = config.get("models", {})
            soniox_config = models_config.get("stt_model", {})
            cartesia_stt_config = models_config.get("cartesia_stt", {})
            
            # Check which STT provider is active (default to Cartesia if both are active or neither)
            soniox_active = soniox_config.get("active", False)
            cartesia_active = cartesia_stt_config.get("active", True)  # Default to Cartesia
            
            streaming_stt_client: BaseRealtimeSTT = None
            
            if cartesia_active and not soniox_active:
                # Use Cartesia STT
                logger.info("Using Cartesia STT provider", "AgentHandlerFactory")
                streaming_stt_client = CartesiaRealtimeSTT(
                    api_key=cartesia_stt_config.get("api_key", ""),
                    model_name=cartesia_stt_config.get("model_name", "ink-whisper"),
                    language=cartesia_stt_config.get("language", "en"),
                    sample_rate=cartesia_stt_config.get("sample_rate", 16000),
                )
            elif soniox_active:
                # Use Soniox STT
                logger.info("Using Soniox STT provider", "AgentHandlerFactory")
                streaming_stt_client = SonioxRealtimeSTT(
                    api_key=soniox_config.get("api_key", ""),
                    model_name=soniox_config.get("model_name", "stt-rt-preview-v2"),
                    sample_rate=soniox_config.get("sample_rate", 16000),
                    num_channels=soniox_config.get("num_channels", 1),
                    audio_format=soniox_config.get("audio_format", "pcm_s16le"),
                    language_hints=soniox_config.get("language_hints", ["en"]),
                    enable_language_identification=soniox_config.get("enable_language_identification", True),
                    enable_speaker_diarization=soniox_config.get("enable_speaker_diarization", False),
                    enable_endpoint_detection=soniox_config.get("enable_endpoint_detection", True),
                )
            else:
                # Default to Cartesia if no active provider specified
                logger.warning("No active STT provider found, defaulting to Cartesia", "AgentHandlerFactory")
                streaming_stt_client = CartesiaRealtimeSTT(
                    api_key=cartesia_stt_config.get("api_key", ""),
                    model_name=cartesia_stt_config.get("model_name", "ink-whisper"),
                    language=cartesia_stt_config.get("language", "en"),
                    sample_rate=cartesia_stt_config.get("sample_rate", 16000),
                )

            # Create inference handler
            system_prompt = agent_config.get("system_prompt", "")
            inference_handler = SpeechToSpeechHandler(
                llm_client=llm_client,
                tts_processor=tts_processor,
                session_handler=session_handler,
                streaming_stt_client=streaming_stt_client,
                system_prompt=system_prompt,
                llm_config=llm_config,
            )

            # Create EOS processor
            eos_processor = EosProcessor()

            # Create voice handler
            voice_handler = RealtimeVoiceHandler(
                session_handler=session_handler,
                inferencing_handler=inference_handler,
                eos_handler=eos_processor,
                streaming_stt_client=streaming_stt_client,
            )

            logger.info(f"Created voice handler for agent: {agent_id}", "AgentHandlerFactory")
            return voice_handler

        except Exception as e:
            logger.error(f"Error creating voice handler: {str(e)}", "AgentHandlerFactory", exc_info=True)
            raise

