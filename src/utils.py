import os
import json
from typing import Dict, Tuple
from dotenv import load_dotenv
import logging


def load_environment(env_path: str = '.env') -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path (str): Path to the .env file.
    """
    load_dotenv(env_path)
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
    }


def get_prompt_by_id(prompt_path: str, prompt_id: str) -> Tuple[str, str]:
    """
    Reads a JSON file with prompts, searches for a prompt by the given ID,
    and returns (system_text, user_text) as strings.

    :param prompt_path: Path to the JSON file with prompts.
    :param prompt_id: ID of the required prompt.
    :returns: tuple (system_text, user_text).
    :raises ValueError: If prompt with the given ID is not found.
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = data.get("prompts", [])

    for prompt in prompts:
        if prompt.get("id") == prompt_id:
            system_text = "\n".join(prompt.get("system", []))
            user_text = "\n".join(prompt.get("user", []))
            return system_text, user_text

    raise ValueError(f"Prompt with id '{prompt_id}' not found in {prompt_path}")


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        log_level (str): Logging level (e.g., "INFO", "DEBUG", "ERROR").
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    return logger




