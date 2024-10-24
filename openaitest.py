import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
from typing import Optional

# Load the environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
MODEL = "gpt-3.5-turbo"
TEST_MESSAGE = {"role": "user", "content": "Say this is a test"}

def get_openai_api_key() -> Optional[str]:
    """Retrieve the OpenAI API key from environment variables."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found. Please check your .env file.")
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")
    return openai_api_key

def initialize_openai_client(api_key: str) -> OpenAI:
    """Initialize the OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key)

def test_openai_connection(client: OpenAI) -> None:
    """Test the connection with the OpenAI API."""
    try:
        response = client.chat.completions.create(model=MODEL, messages=[TEST_MESSAGE])
        logger.info("Response: %s", response.choices[0].message.content)
    except openai.AuthenticationError:
        logger.error("Authentication failed. Please check your API key.")
    except openai.APIError as e:
        logger.error("API Error: %s", e)
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)

def main() -> None:
    """Main function to run the script."""
    api_key = get_openai_api_key()
    client = initialize_openai_client(api_key)
    test_openai_connection(client)

if __name__ == "__main__":
    main()