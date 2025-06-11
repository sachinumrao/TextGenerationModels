import os

from dotenv import load_dotenv
from google import genai
from rich import print

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


def get_gemini_response(prompt: str) -> str:
    llm_client = get_llm_client()
    model_id = "gemini-2.0-flash"
    llm_config = get_llm_config()
    response = llm_client.models.generate_content(
        model=model_id, contents=prompt, config=llm_config
    )

    return response


def get_gemini_response_stream(prompt: str) -> str:
    llm_client = get_llm_client()
    model_id = "gemini-2.0-flash"
    llm_config = get_llm_config()
    response = llm_client.models.generate_content_stream(
        model=model_id, contents=prompt, config=llm_config
    )
    return response


def main():

    response_mode = "stream"
    while True:

        if response_mode == "stream":
            # get input from user
            query = input("User: ")
            response = get_gemini_response_stream(query)
            print("Gemini: ", end="")
            for stream in response:
                print(stream.text, end="")
            print("\n")
        else:
            # get input from user
            query = input("User: ")
            response = get_gemini_response(query)
            # display output to user
            print(f"Gemini: {response.text}")


if __name__ == "__main__":
    main()
