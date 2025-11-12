"""
Rich logging utility for toy backend.
"""
import logging
import os
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
class Logger:
    """Rich logger utility with beautiful console output."""
    
    _logger: Optional[logging.Logger] = None
    _console: Optional[Console] = None
    
    def __init__(self):
        if self._logger is None:
            self._setup_logger()
    
    def _setup_logger(self):
        """Setup Rich logger configuration."""
        try:
            # Try to get config, fallback to defaults if not available
            try:
                from app.utils.static_memory_cache import StaticMemoryCache
                log_config = StaticMemoryCache.get_logging_config()
            except (ImportError, AttributeError, KeyError):
                # Fallback to default config if StaticMemoryCache not initialized
                log_config = {
                    "log_level": "INFO",
                    "log_file": "logs/toy_backend.log",
                    "log_file_max_size": 10485760,
                    "log_file_num_backups": 5,
                    "log_console": True,
                    "use_struct_logger": True
                }
            
            # Install rich traceback handler
            try:
                install(show_locals=True)
            except Exception:
                # Fallback if rich traceback installation fails
                pass
            
            # Create Rich console
            try:
                Logger._console = Console()
            except Exception:
                # Fallback to basic console if Rich fails
                Logger._console = None
            
            # Create logger
            Logger._logger = logging.getLogger("toy_backend")
            Logger._logger.setLevel(getattr(logging, log_config["log_level"]))
            
            # Clear existing handlers
            Logger._logger.handlers.clear()
            
            # Rich console handler
            if log_config.get("log_console", True) and Logger._console:
                rich_handler = RichHandler(
                    console=Logger._console,
                    show_time=True,
                    show_path=True,
                    enable_link_path=True,
                    markup=True
                )
                rich_handler.setLevel(getattr(logging, log_config["log_level"]))
                Logger._logger.addHandler(rich_handler)
            
            # File handler (if needed)
            log_file = log_config.get("log_file", "logs/toy_backend.log")
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=log_config.get("log_file_max_size", 10485760),  # 10MB
                    backupCount=log_config.get("log_file_num_backups", 5)
                )
                file_handler.setLevel(getattr(logging, log_config["log_level"]))
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                Logger._logger.addHandler(file_handler)
            
        except Exception as e:
            # Fallback to basic logging
            Logger._logger = logging.getLogger("toy_backend")
            Logger._logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            Logger._logger.addHandler(handler)
            Logger._logger.error(f"Failed to setup Rich logger: {e}")
    
    def get_logger(self) -> logging.Logger:
        """Get the logger instance."""
        return Logger._logger
    
    def get_console(self) -> Console:
        """Get the Rich console instance."""
        return Logger._console
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get logger instance (class method)."""
        if cls._logger is None:
            cls()._setup_logger()  # Ensure logger is set up
        return cls._logger
    
    @classmethod
    def get_console(cls) -> Console:
        """Get console instance (class method)."""
        if cls._console is None:
            cls._console = cls().get_console()
        return cls._console


# Convenience functions
def get_logger() -> logging.Logger:
    """Get logger instance."""
    return Logger.get_logger()


def get_console() -> Console:
    """Get Rich console instance."""
    return Logger.get_console()


def info(message: str, event_name: str = "toy_backend"):
    """Log info message with Rich formatting."""
    get_logger().info(f"[{event_name}] {message}")


def error(message: str, event_name: str = "toy_backend", exc_info=False):
    """Log error message with Rich formatting."""
    get_logger().error(f"[{event_name}] {message}", exc_info=exc_info)


def warning(message: str, event_name: str = "toy_backend"):
    """Log warning message with Rich formatting."""
    get_logger().warning(f"[{event_name}] {message}")


def debug(message: str, event_name: str = "toy_backend"):
    """Log debug message with Rich formatting."""
    get_logger().debug(f"[{event_name}] {message}")


def success(message: str, event_name: str = "toy_backend"):
    """Log success message with Rich formatting."""
    get_logger().info(f"[{event_name}] ✓ {message}")


# Logger wrapper class for direct import
class LoggerWrapper:
    """Wrapper class to provide logger-like interface."""
    
    def info(self, message: str, event_name: str = "toy_backend"):
        """Log info message."""
        get_logger().info(f"[{event_name}] {message}")
    
    def error(self, message: str, event_name: str = "toy_backend", exc_info=False):
        """Log error message."""
        get_logger().error(f"[{event_name}] {message}", exc_info=exc_info)
    
    def warning(self, message: str, event_name: str = "toy_backend"):
        """Log warning message."""
        get_logger().warning(f"[{event_name}] {message}")
    
    def debug(self, message: str, event_name: str = "toy_backend"):
        """Log debug message."""
        get_logger().debug(f"[{event_name}] {message}")


# Export logger instance for direct import
logger = LoggerWrapper()

