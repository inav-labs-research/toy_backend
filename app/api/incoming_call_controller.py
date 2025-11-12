"""
Incoming call controller with WebSocket endpoint.
"""
import uuid
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter
from app.agents.agent_loader import AgentLoader
from app.factories.agent_handler_factory import AgentHandlerFactory
from app.media_stream_handler.web_call_session_handler import WebCallSessionHandler
from app.media_stream_handler.websocket_stream_handler import WebSocketStreamHandler
from app.utils.logger import logger

router = APIRouter()


@router.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle the media stream connection from website."""
    request_id = str(uuid.uuid4())
    
    # Get agent_id from query parameters
    agent_id = websocket.query_params.get("agent_id", "harry_potter")
    logger.info(f"Incoming call websocket request arrived for agent: {agent_id}", "incoming_call_controller")

    try:
        # Get agent config
        agent_config = AgentLoader.get_agent(agent_id)
        if not agent_config:
            logger.error(f"Agent {agent_id} not found", "incoming_call_controller")
            await websocket.close(code=1008, reason=f"Agent {agent_id} not found")
            return

        # Get first response message
        first_response_message = agent_config.get("first_response_message")

        # Create stream handler
        stream_handler = WebSocketStreamHandler()

        # Create session handler
        session_handler = WebCallSessionHandler(
            websocket=websocket,
            stream_handler=stream_handler,
            request_id=request_id,
            user_id="default_user",
            first_response_message=first_response_message,
        )

        # Create voice handler
        real_time_voice_handler = await AgentHandlerFactory.create_voice_handler_for_agent(
            agent_id=agent_id,
            session_handler=session_handler,
        )

        # Handle stream
        await stream_handler.handle_stream(websocket, real_time_voice_handler)

    except WebSocketDisconnect:
        logger.error("Media stream disconnected", "incoming_call_controller")
    except Exception as e:
        logger.error(f"Error in media stream: {str(e)}", "incoming_call_controller", exc_info=True)
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except:
            pass

