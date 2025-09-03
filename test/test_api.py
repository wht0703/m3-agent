import json
from openai import OpenAI

CLIENTS = {}
TEST_PROMPT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize AI in one sentence."},
]
# api utils
with open("configs/api_config.json") as f:
    api_config = json.load(f)
    for model in api_config.keys():
        client = OpenAI(api_key=api_config[model]["api_key"])
        # Add base url in case of custom gemini endpoint
        if model == "gemini-1.5-pro-002":
            client.base_url = api_config[model]["base_url"]
        CLIENTS[model] = client    

def test_gemini_connection():
    response = CLIENTS["gemini-1.5-pro-002"].chat.completions.create(
        model = "gemini-1.5-pro-002",
        messages = TEST_PROMPT
    )
    return response.choices[0].message.content

def test_gpt_connection():
    response = CLIENTS["gpt-4o-2024-11-20"].chat.completions.create(
        model = "gpt-4o-2024-11-20",
        messages = TEST_PROMPT
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    print("Gemini response:", test_gemini_connection())
    # print("GPT response:", test_gpt_connection())