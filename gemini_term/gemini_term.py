import datetime
import json
import os

from dotenv import load_dotenv
from google import genai
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()


def get_llm_client():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)
    return client


def get_llm_config():
    llm_config = {
        "system_instruction": "Your job is to provide concise and grounded answers for questions.",
        "temperature": 0.7,
    }

    return llm_config


def get_gemini_chat_response_stream(prompt: str) -> str:
    llm_client = get_llm_client()
    model_id = "gemini-2.0-flash"
    llm_config = get_llm_config()
    response = llm_client.models.generate_content_stream(
        model=model_id, contents=prompt, config=llm_config
    )

    return response


def main():

    message_history = []
    chat_start_timestamp = str(datetime.datetime.now()).reaplce(" ", "_")

    try:
        while True:
            # get input from user
            query = input("User: ")
            log_resp = ""
            response = get_gemini_chat_response_stream(query)
            print("Gemini: ", end="")
            for stream in response:
                output_chunk = stream.text
                print(output_chunk, end="")
                log_resp += output_chunk
            print("\n")

    except KeyboardInterrupt:
        # check if logs directory is present
        logs_dir = "./logs"
        logs_file_name = f"chta_logs_{chat_start_timestamp}.json"

        # dump logs file
        # with open(os.path.join(logs_dir, logs_file_name), "w") as f:
        #     json.dumps(message_history)


if __name__ == "__main__":
    main()

# TODO:
# - add context length and max-tokens settings in llm_config
# - add rich display of markdown
# - add system prompt from user
