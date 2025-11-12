"""
Gemini Prediction Client for Google's Gemini API
"""
import asyncio
import json
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.models.language_models.online_serving.model_streaming_response import ModelStreamingResponse
from app.utils.logger import logger


class GeminiPredictionClient:
    """Client for Google's Gemini API prediction."""
    
    def __init__(self, config: dict):
        """Initialize Gemini prediction client with configuration."""
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "gemini-2.0-flash")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.api_base = config.get("api_base", "https://generativelanguage.googleapis.com/v1beta")
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        logger.info(f"GeminiPredictionClient: Initialized with model {self.model_name}", "gemini_prediction_client")
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a single response from Gemini API."""
        try:
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini_format(messages)
            
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                    "topP": 0.8,
                    "topK": 10
                }
            }

            logger.info(f"GeminiPredictionClient: Payload: {payload}", "gemini_prediction_client")
            url = f"{self.api_base}/models/{self.model_name}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            text_parts = []
                            for part in candidate["content"]["parts"]:
                                if "text" in part:
                                    text_parts.append(part["text"])
                            return "".join(text_parts)
                    
                    logger.warning("GeminiPredictionClient: No valid response content found", "gemini_prediction_client")
                    return "I'm sorry, I couldn't generate a response at this time."
                else:
                    logger.error(f"GeminiPredictionClient: API request failed with status {response.status_code}: {response.text}", "gemini_prediction_client")
                    return "I'm sorry, there was an error processing your request."
                    
        except Exception as e:
            logger.error(f"GeminiPredictionClient: Error generating response: {str(e)}", "gemini_prediction_client", exc_info=True)
            return "I'm sorry, there was an error processing your request."
    
    async def generate_response_streaming(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Gemini API."""
        try:
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini_format(messages)
            
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                    "topP": 0.8,
                    "topK": 10
                }
            }
            
            logger.info(f"GeminiPredictionClient: Streaming payload: {payload}", "gemini_prediction_client")
            url = f"{self.api_base}/models/{self.model_name}:streamGenerateContent"
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    logger.info(f"GeminiPredictionClient: Streaming response status: {response.status_code}", "gemini_prediction_client")
                    
                    if response.status_code == 200:
                        chunk_count = 0
                        logger.info("GeminiPredictionClient: Starting to read streaming response...", "gemini_prediction_client")
                        async for line in response.aiter_lines():
                            if line.strip():
                                logger.debug(f"GeminiPredictionClient: Received line: {line}", "gemini_prediction_client")
                                try:
                                    # Parse streaming response - Gemini uses different format
                                    if line.startswith("data: "):
                                        data = line[6:]  # Remove "data: " prefix
                                        if data.strip() == "[DONE]":
                                            logger.info("GeminiPredictionClient: Stream completed", "gemini_prediction_client")
                                            break
                                        
                                        chunk = json.loads(data)
                                        logger.debug(f"GeminiPredictionClient: Parsed chunk: {chunk}", "gemini_prediction_client")
                                        
                                        # Check for candidates in the chunk
                                        if "candidates" in chunk and len(chunk["candidates"]) > 0:
                                            candidate = chunk["candidates"][0]
                                            logger.debug(f"GeminiPredictionClient: Candidate: {candidate}", "gemini_prediction_client")
                                            
                                            # Check for content in candidate
                                            if "content" in candidate and "parts" in candidate["content"]:
                                                for part in candidate["content"]["parts"]:
                                                    if "text" in part and part["text"]:
                                                        chunk_count += 1
                                                        logger.debug(f"GeminiPredictionClient: Yielding text chunk {chunk_count}: {part['text']}", "gemini_prediction_client")
                                                        yield part["text"]
                                            # Check for finish reason
                                            elif "finishReason" in candidate:
                                                logger.info(f"GeminiPredictionClient: Finish reason: {candidate['finishReason']}", "gemini_prediction_client")
                                                break
                                    else:
                                        # Try to parse as direct JSON (some responses might not have "data: " prefix)
                                        try:
                                            chunk = json.loads(line)
                                            logger.debug(f"GeminiPredictionClient: Direct JSON chunk: {chunk}", "gemini_prediction_client")
                                            
                                            if "candidates" in chunk and len(chunk["candidates"]) > 0:
                                                candidate = chunk["candidates"][0]
                                                if "content" in candidate and "parts" in candidate["content"]:
                                                    for part in candidate["content"]["parts"]:
                                                        if "text" in part and part["text"]:
                                                            chunk_count += 1
                                                            logger.debug(f"GeminiPredictionClient: Yielding direct text chunk {chunk_count}: {part['text']}", "gemini_prediction_client")
                                                            yield part["text"]
                                        except json.JSONDecodeError:
                                            continue
                                            
                                except json.JSONDecodeError as e:
                                    logger.debug(f"GeminiPredictionClient: JSON decode error: {e}", "gemini_prediction_client")
                                    continue
                                except Exception as e:
                                    logger.debug(f"GeminiPredictionClient: Error parsing streaming chunk: {e}", "gemini_prediction_client")
                                    continue
                        
                        logger.info(f"GeminiPredictionClient: Stream completed with {chunk_count} chunks", "gemini_prediction_client")
                        if chunk_count == 0:
                            logger.warning("GeminiPredictionClient: No text chunks received from streaming API, falling back to batch API", "gemini_prediction_client")
                            # Fallback to batch API if streaming returns no chunks
                            batch_response = await self.generate_response(messages)
                            if batch_response:
                                yield batch_response
                    else:
                        logger.error(f"GeminiPredictionClient: Streaming API request failed with status {response.status_code}", "gemini_prediction_client")
                        error_text = await response.aread()
                        logger.error(f"GeminiPredictionClient: Error response: {error_text}", "gemini_prediction_client")
                        yield "I'm sorry, there was an error processing your request."
                        
        except Exception as e:
            logger.error(f"GeminiPredictionClient: Error in streaming response: {str(e)}", "gemini_prediction_client", exc_info=True)
            yield "I'm sorry, there was an error processing your request."
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert messages to Gemini API format."""
        contents = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # Gemini doesn't have a system role, so we prepend it to the first user message
                if contents and contents[-1].get("role") == "user":
                    # Prepend system message to existing user message
                    existing_text = contents[-1]["parts"][0]["text"]
                    contents[-1]["parts"][0]["text"] = f"{content}\n\n{existing_text}"
                else:
                    # Create a new user message with system content
                    contents.append({
                        "role": "user",
                        "parts": [{"text": content}]
                    })
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
        
        return contents
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "gemini"
        }

