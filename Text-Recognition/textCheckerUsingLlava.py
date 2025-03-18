import requests
import json


def check_text_via_llava(text, model="llava:13b", host="http://localhost:11434"):
    """Checks and corrects the give text using the LLaVA model runing on Ollama and returns the response."""

    url = f"{host}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": "Clean and correct the text, return only the corrected text and nothing else : "
        + text,
        "stream": False,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json().get("response", "No response received.")
    else:
        return f"Sorry, cannot process the request right now!"


if __name__ == "__main__":
    text = """"""
    response = check_text_via_llava(text)
    print(response)
