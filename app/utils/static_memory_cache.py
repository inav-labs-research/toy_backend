"""
Static memory cache for configuration and models.
"""
import json
from typing import Dict, Any


class StaticMemoryCache:
    """Static class to store config and models in memory."""
    config: Dict[str, Any] = {}
    models: Dict[str, Any] = {}
    noise_reduction_pipeline = None
    vad_model = None

    @classmethod
    def initialize(cls, config_file: str = "config.json"):
        """Load config and models into memory at startup."""
        with open(config_file, "r") as f:
            cls.config = json.load(f)
        cls._initialize_noise_reduction_pipeline()
        cls._initialize_vad_model()

    @classmethod
    def _initialize_noise_reduction_pipeline(cls):
        """Initialize noise reduction pipeline and store it in memory."""
        noise_reduction_config = cls.config.get("noise_reduction", {})
        if noise_reduction_config and noise_reduction_config.get("should_use_noise_reduction"):
            # Noise reduction can be added later if needed
            cls.noise_reduction_pipeline = None
        else:
            cls.noise_reduction_pipeline = None

    @classmethod
    def get_config(cls, section: str, key: str = None):
        """Retrieve configuration value from the static memory cache."""
        if key:
            return cls.config.get(section, {}).get(key)
        return cls.config.get(section, {})

    @classmethod
    def get_model(cls, model_name: str):
        """Retrieve model from the static memory cache."""
        return cls.models.get(model_name)

    @classmethod
    def _initialize_vad_model(cls):
        """Initialize VAD model."""
        try:
            vad = cls.config.get("models", {}).get("vad_model", {})
            if vad:
                # Lazy import torch to avoid import errors
                import torch
                cls.vad_model = torch.hub.load(
                    repo_or_dir=vad.get("model_repo", "snakers4/silero-vad"),
                    model=vad.get("model_name", "silero_vad"),
                    force_reload=False
                )
        except Exception as e:
            from app.utils.logger import error
            error(f"Failed to initialize VAD model: {e}", "static_memory_cache")
            cls.vad_model = None

    @classmethod
    def get_vad_model(cls):
        """Get VAD model."""
        return cls.vad_model

    @classmethod
    def get_noise_reduction_pipeline(cls):
        """Retrieve noise cancellation pipeline."""
        return cls.noise_reduction_pipeline

    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration."""
        return cls.config.get("logging", {
            "log_level": "INFO",
            "log_file": "logs/toy_backend.log",
            "log_file_max_size": 10485760,
            "log_file_num_backups": 5,
            "log_console": True,
            "use_struct_logger": True
        })

