"""
Text transformation utilities.
"""
import re
from typing import List


class TextTransformation:
    """Utility class for text transformation operations"""

    @staticmethod
    def clean_text_for_speech(text: str) -> str:
        """
        Clean and normalize the text for TTS:
        - Remove reasoning/thinking tags (<think>, <think>, etc.)
        - Remove emojis and formatting characters
        - Fix broken words with spaces
        - Normalize spacing around punctuation and hyphens
        """
        # Remove reasoning/thinking tags first (including redacted_reasoning)
        # Handle both opening and closing tags, and content between them
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Also remove standalone reasoning markers (opening or closing tags without pairs)
        text = re.sub(r'</?redacted_reasoning>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
        
        # Remove unwanted formatting characters
        for ch in ['*', '`', '~', '^', '|']:
            text = text.replace(ch, ' ')

        # Remove emojis
        emoji_pattern = re.compile(
            "[" +
            u"\U0001F600-\U0001F64F" +  # emoticons
            u"\U0001F300-\U0001F5FF" +
            u"\U0001F680-\U0001F6FF" +
            u"\U0001F1E0-\U0001F1FF" +
            u"\u2600-\u26FF" +
            u"\u2700-\u27BF" +
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)

        # Collapse multiple spaces to single
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

