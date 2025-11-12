"""
End of speech processor.
"""
import numpy as np
from app.utils.static_memory_cache import StaticMemoryCache
from app.utils.logger import logger


class EosProcessor:
    """End of speech detection processor."""
    
    def __init__(
        self,
        max_allowed_silence_duration: float = 1.0,
        vad_processor=None,
    ):
        self.max_allowed_silence_duration = max_allowed_silence_duration
        self.vad_processor = vad_processor
        self.vad_model = StaticMemoryCache.get_vad_model()
        self.sampling_rate = 16000
        self.silence_threshold = 0.01  # Simple energy-based threshold

    def is_end_of_speech_detected(self, audio_samples: np.ndarray):
        """
        Detect end of speech.
        Returns: (is_eos_detected, voice_detected_in_frame)
        """
        if len(audio_samples) == 0:
            return False, False

        # Simple energy-based VAD
        audio_energy = np.mean(np.abs(audio_samples))
        voice_detected = audio_energy > self.silence_threshold

        # Check if we have enough silence
        if not voice_detected:
            # Check last portion of audio for silence
            if len(audio_samples) > self.sampling_rate * self.max_allowed_silence_duration:
                last_portion = audio_samples[-int(self.sampling_rate * self.max_allowed_silence_duration):]
                last_energy = np.mean(np.abs(last_portion))
                if last_energy < self.silence_threshold:
                    return True, False

        return False, voice_detected

