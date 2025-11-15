"""
Gemini Prediction Client for Google's Gemini API using official google-genai library
"""
import asyncio
import threading
from typing import List, Dict, Any, Optional, AsyncGenerator
from google import genai
from app.models.language_models.online_serving.model_streaming_response import ModelStreamingResponse
from app.utils.logger import logger


class GeminiPredictionClient:
    """Client for Google's Gemini API prediction using official google-genai library."""
    
    def __init__(self, config: dict):
        """Initialize Gemini prediction client with configuration."""
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "gemini-2.0-flash")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.api_base = config.get("api_base", "https://generativelanguage.googleapis.com/v1beta")
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        # Initialize the official Google genai client
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(f"GeminiPredictionClient: Initialized with model {self.model_name}", "gemini_prediction_client")
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a single response from Gemini API."""
        try:
            # Convert messages to contents format for genai library
            contents = self._convert_messages_to_genai_contents(messages)
            
            # Don't log full contents (contains system prompt) - only log that request is being made
            logger.debug(f"GeminiPredictionClient: Sending batch request to {self.model_name}", "gemini_prediction_client")
            
            # Use the official genai client for batch generation
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.8,
                    "top_k": 10
                }
            )
            
            # Extract text from response
            if response and hasattr(response, 'text') and response.text:
                return response.text
            elif response and hasattr(response, 'candidates') and response.candidates:
                # Fallback: extract from candidates if text attribute not available
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return "".join(text_parts)
            
            logger.warning("GeminiPredictionClient: No valid response content found", "gemini_prediction_client")
            return "I'm sorry, I couldn't generate a response at this time."
                    
        except Exception as e:
            logger.error(f"GeminiPredictionClient: Error generating response: {str(e)}", "gemini_prediction_client", exc_info=True)
            return "I'm sorry, there was an error processing your request."
    
    async def generate_response_streaming(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Generate a streaming response from Gemini API using official genai library."""
        try:
            # Convert messages to contents format for genai library
            contents = self._convert_messages_to_genai_contents(messages)
            
            # Don't log full contents (contains system prompt) - only log that streaming request is being made
            logger.debug(f"GeminiPredictionClient: Starting streaming request to {self.model_name}", "gemini_prediction_client")
            
            # The genai library's generate_content_stream returns a synchronous iterator
            # We need to iterate it in a thread to make it async-compatible
            loop = asyncio.get_event_loop()
            
            # Get the streaming response iterator (this returns immediately)
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.8,
                    "top_k": 10
                }
            )
            
            chunk_count = 0
            logger.info("GeminiPredictionClient: Starting to read streaming response...", "gemini_prediction_client")
            
            # Process chunks one by one in executor to maintain streaming behavior
            # Use a queue to pass chunks from sync iterator to async generator
            chunk_queue = asyncio.Queue()
            finished = False
            
            def _process_chunks():
                """Process chunks from the synchronous iterator and put them in queue."""
                nonlocal finished
                try:
                    for chunk in response:
                        if hasattr(chunk, 'text') and chunk.text:
                            # Put chunk in queue (will be processed async)
                            asyncio.run_coroutine_threadsafe(chunk_queue.put(chunk.text), loop)
                        elif hasattr(chunk, 'candidates') and chunk.candidates:
                            # Fallback: extract text from candidates if text attribute not available
                            for candidate in chunk.candidates:
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            asyncio.run_coroutine_threadsafe(chunk_queue.put(part.text), loop)
                                # Check for finish reason
                                if hasattr(candidate, 'finish_reason'):
                                    finish_reason = candidate.finish_reason
                                    if finish_reason in ["STOP", "MAX_TOKENS"]:
                                        logger.info(f"GeminiPredictionClient: Finish reason: {finish_reason}, stream complete", "gemini_prediction_client")
                                        finished = True
                                        break
                        if finished:
                                            break
                except Exception as e:
                    logger.error(f"GeminiPredictionClient: Error processing chunks: {str(e)}", "gemini_prediction_client", exc_info=True)
                finally:
                    # Signal completion
                    asyncio.run_coroutine_threadsafe(chunk_queue.put(None), loop)
                                            
            # Start processing chunks in background thread
            thread = threading.Thread(target=_process_chunks, daemon=True)
            thread.start()
            
            # Yield chunks as they arrive from the queue
            while True:
                text = await chunk_queue.get()
                if text is None:  # Signal that streaming is complete
                    break
                chunk_count += 1
                logger.debug(f"GeminiPredictionClient: Yielding text chunk {chunk_count}: {text[:50]}...", "gemini_prediction_client")
                yield text
            
            logger.info(f"GeminiPredictionClient: Stream completed with {chunk_count} chunks", "gemini_prediction_client")
            if chunk_count == 0:
                logger.warning("GeminiPredictionClient: No text chunks received from streaming API, falling back to batch API", "gemini_prediction_client")
                # Fallback to batch API if streaming returns no chunks
                batch_response = await self.generate_response(messages)
                if batch_response:
                    yield batch_response
                        
        except Exception as e:
            logger.error(f"GeminiPredictionClient: Error in streaming response: {str(e)}", "gemini_prediction_client", exc_info=True)
            yield "I'm sorry, there was an error processing your request."
    
    def _convert_messages_to_genai_contents(self, messages: List[Dict[str, str]]) -> List[str]:
        """
        Convert messages to contents format for genai library.
        The genai library expects a list of strings or a list of Content objects.
        For simplicity, we'll convert to a list of strings with role prefixes.
        """
        contents = []
        system_prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # Accumulate system prompt to prepend to first user message
                if system_prompt:
                    system_prompt += "\n\n" + content
                else:
                    system_prompt = content
            elif role == "user":
                # Prepend system prompt to first user message if exists
                user_content = f"{system_prompt}\n\n{content}" if system_prompt else content
                contents.append(user_content)
                system_prompt = ""  # Clear after using
            elif role == "assistant":
                # For assistant messages, we need to handle them differently
                # The genai library handles conversation history automatically
                # For now, we'll include them as part of the conversation
                contents.append(content)
        
        # If system prompt wasn't used, prepend to first content
        if system_prompt and contents:
            contents[0] = f"{system_prompt}\n\n{contents[0]}"
        elif system_prompt and not contents:
            # Only system message, create a user message
            contents.append(system_prompt)
        
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

