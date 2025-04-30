import logging
import subprocess
import ollama
import speech_recognition as sr
import subprocess
import json
import os
from gtts import gTTS
import vosk
import requests

######
import base64
import requests

# Configure logging
logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

WAKEUP_WORD = "computer"
HOST_IP = os.getenv("HOST_IP")
OLLAMA_HOST_URL = f"http://{HOST_IP}:11434/api/generate"


def text_to_speech(text, language='en', slow=False):
    """
    Convert text to speech and play it.
    
    Parameters:
    - text (str): The text to convert to speech.
    - language (str): Language code (default is 'en' for English).
    - slow (bool): Whether to slow down the speech (default is False).
    """
    # Create gTTS object
    speech = gTTS(text=text, lang=language, slow=slow)
    
    # Define audio file name
    audio_file = "output.mp3"
    
    # Save the audio file
    speech.save(audio_file)
    
    # Play the audio file
    subprocess.run(["mpg321", "-q", audio_file])  # -q for quiet mode
    
    # Remove the file after playing to avoid clutter
    os.remove(audio_file)
    logging.info("Text to speech conversion complete")

def start_object_detection():
    logging.info("Starting Object Detection")
    # return subprocess.Popen(["python3", "objectdetection/detection.py", "-i", "rpi"])
    return subprocess.Popen(["python3", "detection_depth/app2.py","--input", "rpi", "--apps_infra_path", "/home/jenil/Documents/hailo-rpi5-examples/hailo-apps-infra/"])


def run_text_recognition():
    logging.info("Running Text Recognition")
    # result = subprocess.run(
    #     ["python3", "textrecognition/text_recognition.py"], 
    #     capture_output=True, 
    #     text=True
    # )
    # return result.stdout.strip()  # Capture cleaned text output
    file_path = os.path.join(os.getcwd(), "images", "text.jpg")
    print(file_path)
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    response = requests.post(
        f"http://{HOST_IP}:8000/upload-image/", json={"image_base64": encoded_string}
    )

    if response.status_code == 200:
        result = response.json()
    return result['text']


def run_scenic_understanding():
    logging.info("Running Scenic Understanding")
    result = subprocess.run(
        ["python3", "scenicunderstanding/scenic_understanding.py"],
        capture_output = True,
        text = True
    )
    return result.stdout.strip()


def get_audio_input(wait_for_wake_word: bool = True) -> str:
    """
    Captures audio from the microphone and converts it to text using Vosk.
    If wait_for_wake_word is True, listens for the wake word "hey computer" before capturing the command.

    Args:
        wait_for_wake_word (bool): Whether to wait for a wake word before recognizing the command.

    Returns:
        str: The recognized command text, or an empty string if recognition fails.
    """
    # initialize vosk model (transcriber)
    model = vosk.Model("model/vosk-model-small-en-us-0.15")
    logging.info("Inside vosk")
    # setup speechrecognition recognizer and microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    logging.info("setup mic")
    # If waiting for wake word, continuously listen until it's detected
    if wait_for_wake_word:
        logging.info("Waiting for wake word...")
        while True:
            logging.info("Entered voice infinite loop")
            with microphone as source:
                audio = recognizer.listen(source)
            # convert audio to .WAV format for vosk
            wav_data = audio.get_wav_data()
            # create a vosk recognizer with the model's sample rate
            rec = vosk.KaldiRecognizer(model, audio.sample_rate)
            # Process the audio
            rec.AcceptWaveform(wav_data)
            # Parse the JSON result to get the text
            result = json.loads(rec.Result())
            text = result.get("text", "")
            # check if the WAKEUP_WORD is in the recognized text
            if WAKEUP_WORD in text.lower():
                logging.info("wakeup word detected...")
                break
            logging.info(f"got wakeup word: {text}")

    text_to_speech("Yes?")
    # capture the actual command
    logging.info("Listening for command...")
    with microphone as source:
        audio = recognizer.listen(source)
    # Convert audio to WAV data
    wav_data = audio.get_wav_data()
    # Create a Vosk recognizer
    rec = vosk.KaldiRecognizer(model, audio.sample_rate)
    # Process the audio
    rec.AcceptWaveform(wav_data)
    # Parse the result
    result = json.loads(rec.Result())
    # Extract the command text, default to empty string if not found
    command = result.get("text", "")
    logging.info(f"text recognized: {command}")
    return command

def analyze_intent(command):
    logging.info("Analyzing intent")
    prompt = (
        f"Classify the intent of this command: '{command}'. "
        f"Respond with ONLY one of the following words: 'text_recognition', 'scenic_understanding', or 'unknown'. "
        f"Do not provide any explanation."
    )
    
    payload = {
    "model": "mistral",
    "prompt": prompt,
    "stream": False
    }
    response = requests.post(OLLAMA_HOST_URL, json=payload)
    logging.info("Got response")
    result = response.json()
    intent = result["response"].strip().lower()
    # intent = response["message"]["content"].strip().lower()
    logging.info(f"Intent detected: {intent}")

    return intent

def capture_image(feature: str):
    if feature == "text_recognition":
        image_path = "images/text.jpg"
    else:
        image_path = "images/scene.png"
    try:
        if os.name == "posix":
            if os.uname().sysname == "Darwin":
                subprocess.run(["imagesnap", "-q", image_path], check=True)
            else:
                subprocess.run(
                    # ["fswebcam", "-r", "1280x720", "--no-banner", image_path],
                    ["libcamera-still", "-o", image_path],
                    check=True,
                )
        else:
            raise OSError("Unsupported OS for image capture.")

        logging.info(f"Image captured and saved as {image_path}")
        return image_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to capture image: {e}")
        return None
    except OSError as e:
        logging.error(f"OS Error: {e}")
        return None

def main():
    # logging.info("getting audio")
    # command = get_audio_input(wait_for_wake_word=True)
    # logging.info("Out of main loop")
    process = start_object_detection()
    logging.info("test")
    text_to_speech("Object Detection Started")
    while True:
        logging.info("Entered Infinite loop")
        # get command
        command = get_audio_input(wait_for_wake_word=True)
        logging.info("got command")
        # get intention of command
        intent = analyze_intent(command)

        if intent == "text_recognition":
            text_to_speech("Recognizing Text")
            process.terminate()
            capture_image(intent)
            cleaned_text = run_text_recognition()
            text_to_speech(cleaned_text)
        elif intent == "scenic_understanding":
            text_to_speech("Wait a moment")
            process.terminate()
            capture_image(intent)
            scene_description = run_scenic_understanding()
            text_to_speech(scene_description)
        else:
            text_to_speech("Sorry, I could not understand. Can you please repeat?")
            logging.warning(f"Unkown intent received: {intent}")

        process = start_object_detection()
        logging.info("Restarted object detection")


if __name__ == "__main__":
    main()
