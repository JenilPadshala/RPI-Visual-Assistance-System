import requests
import json
import base64
import logging
import os
# Configure logging
logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

HOST_IP = os.getenv("HOST_IP")
OLLAMA_HOST_URL = f"http://{HOST_IP}:11434/api/generate"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_text(image_path, model="llava:13b", host_url=OLLAMA_HOST_URL):
    """Extracts text from the given image"""
    headers = {"Content-Type": "application/json"}

    if image_path:
        image = encode_image(image_path)
    else:
        return "Please provide an image"
    logging.info("Image encoded successfully")
    data = {
        "model": model,
        "prompt": "This is what is in front of me. Describe the scene. Keep it short, less than twenty words. Refer to is as scene.",
        "stream": False,
        "images": [image],
    }

    response = requests.post(host_url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json().get("response", "No response received.")
    else:
        return f"Sorry, cannot process the request right now!"


if __name__ == "__main__":
    image_path = os.path.join(os.getcwd(), "images", "scene.png")
    response = extract_text(image_path)
    logging.info(f"Scene described: {response}")
    print(response)
