"""
Model streaming response data class.
"""
from dataclasses import dataclass
from typing import Optional, List, Any


@dataclass
class ModelStreamingResponse:
    """Response from streaming prediction."""
    text: str
    is_tool_call_response: bool = False
    tool_calls: Optional[List[Any]] = None

