"""
Main entry point for toy_backend server.
"""
import signal
import sys
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import asynccontextmanager
from app.api.incoming_call_controller import router as incoming_call_router
from app.utils.static_memory_cache import StaticMemoryCache
from app.agents.agent_loader import AgentLoader
from app.utils.logger import logger, info

# Global server instance for shutdown
_server = None


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    _shutdown_called = False
    
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        nonlocal _shutdown_called
        if _shutdown_called:
            # Second Ctrl+C - force exit immediately
            info("Force exit (second Ctrl+C)", "shutdown")
            sys.exit(1)
        
        _shutdown_called = True
        info(f"Received signal {signum}, initiating graceful shutdown...", "shutdown")
        
        if _server:
            try:
                # Stop the uvicorn server immediately
                _server.should_exit = True
            except Exception as e:
                info(f"Error during shutdown: {e}", "shutdown")
        
        # Force exit after short delay if server doesn't stop
        import threading
        def force_exit():
            import time
            time.sleep(2.0)  # Give 2 seconds for graceful shutdown
            if _server and hasattr(_server, 'should_exit') and _server.should_exit:
                info("Force exiting after timeout...", "shutdown")
                sys.exit(0)
        threading.Thread(target=force_exit, daemon=True).start()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def _shutdown_all_connections():
    """Shutdown all active websocket connections."""
    # This will be called from the server's event loop
    # All websockets should check should_stop and close themselves
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Setup signal handlers
    setup_signal_handlers()
    
    # Startup
    try:
        # Initialize static memory cache
        StaticMemoryCache.initialize("config.json")
        info("StaticMemoryCache initialized", "startup")
        
        # Initialize agent loader
        AgentLoader.initialize("agents.json")
        info("AgentLoader initialized", "startup")
        
        info("Server startup complete", "startup")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", "startup", exc_info=True)
        raise

    yield

    # Shutdown
    info("Server shutting down", "shutdown")


# Create FastAPI app
app = FastAPI(
    title="Toy Backend",
    description="Simple voice agent backend with Harry Potter agent",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(incoming_call_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Toy Backend API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/api/media-stream?agent_id=harry_potter"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    # Initialize config first (for direct execution)
    try:
        StaticMemoryCache.initialize("config.json")
        AgentLoader.initialize("agents.json")
    except:
        pass  # Already initialized by lifespan
    port = StaticMemoryCache.get_config("server", "port") or 5050
    
    # Setup signal handlers before starting server
    setup_signal_handlers()
    
    # Create server config
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    _server = uvicorn.Server(config)
    
    try:
        _server.run()
    except KeyboardInterrupt:
        info("Keyboard interrupt received, shutting down...", "shutdown")
    except Exception as e:
        info(f"Server error: {e}", "shutdown")
        raise
    finally:
        info("Server stopped", "shutdown")

