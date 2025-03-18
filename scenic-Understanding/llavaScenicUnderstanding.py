import requests
import json
import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_text(image_path, model="llava:13b", host="http://localhost:11434"):
    """Extracts text from the given image"""
    url = f"{host}/api/generate"
    headers = {"Content-Type": "application/json"}

    if image_path:
        image = encode_image(image_path)
    else:
        return "Please provide an image"

    data = {
        "model": model,
        "prompt": "This is what is in front of me. Describe the scene. Keep it short. Refer to is as scene.",
        "stream": False,
        "images": [image],
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json().get("response", "No response received.")
    else:
        return f"Sorry, cannot process the request right now!"


if __name__ == "__main__":
    image_path = ""
    response = extract_text(image_path)
    print(response)
