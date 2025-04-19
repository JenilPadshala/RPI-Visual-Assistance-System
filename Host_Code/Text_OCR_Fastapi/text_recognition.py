import cv2
import numpy as np
import easyocr
import string
import re
from spellchecker import SpellChecker
import requests
import json
import logging
import os

HOST_IP = "0.0.0.0"  # os.getenv("HOST_IP")
OLLAMA_HOST_URL = f"http://{HOST_IP}:11434/api/generate"

# Configure logging
logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize spell checker
spell = SpellChecker()


def load_image(image_path):
    """Load an image from the given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image


def preprocess_image(image):
    """Convert to grayscale, blur, and apply adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def detect_lines(thresh):
    """Detect lines in the thresholded image using Hough Transform."""
    lines = cv2.HoughLinesP(
        thresh, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )
    return lines


def calculate_skew_angle(lines):
    """Calculate the skew angle based on detected lines."""
    if lines is not None:
        angles = []
        weights = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            if -45 <= angle <= 45:
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angles.append(angle)
                weights.append(length)

        if angles:
            sum_sin = sum(w * np.sin(a * np.pi / 180) for a, w in zip(angles, weights))
            sum_cos = sum(w * np.cos(a * np.pi / 180) for a, w in zip(angles, weights))
            skew_angle = np.arctan2(sum_sin, sum_cos) * 180 / np.pi
        else:
            skew_angle = 0
    else:
        skew_angle = 0
    return skew_angle


def deskew_image(image, skew_angle):
    """Deskew the image based on the calculated skew angle."""
    if skew_angle != 0:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        deskewed = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    else:
        deskewed = image
    return deskewed


def clean_text(text_list):
    """
    Clean and reconstruct text blocks to correct OCR errors and improve coherence.

    Args:
        text_list (list): List of raw text blocks from EasyOCR.

    Returns:
        str: Cleaned and reconstructed text.
    """
    # Join all text blocks into a single string
    raw_text = " ".join(text_list)

    # Step 1: Remove non-printable characters
    text = "".join(char for char in raw_text if char in string.printable)

    # Step 2: Normalize whitespace and remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Step 3: Fix split words (e.g., "Moti vated" → "motivated")
    words = text.split()
    corrected_words = []
    i = 0
    while i < len(words):
        # Check if current and next word might be split parts of one word
        if i + 1 < len(words) and len(words[i]) <= 4 and not words[i].endswith("."):
            combined = words[i] + words[i + 1]
            if combined in spell or spell.correction(combined) == combined:
                corrected_words.append(combined)
                i += 2
            else:
                corrected_words.append(words[i])
                i += 1
        else:
            corrected_words.append(words[i])
            i += 1

    # Step 4: Apply spell checking to each word
    corrected_text = []
    for word in corrected_words:
        if word in spell or word.isdigit() or word.endswith((".", ",", "!", "?")):
            corrected_text.append(word)  # Keep if known, number, or punctuated
        else:
            correction = spell.correction(word)
            corrected_text.append(correction if correction else word)

    # Step 5: Reconstruct text with basic sentence structure
    text = " ".join(corrected_text)
    # Capitalize first letter and add period if missing
    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith((".", "!", "?")):
            text += "."

    return text


def extract_text(image):
    """Extract text from the image using EasyOCR."""
    reader = easyocr.Reader(["en"])  # English language
    result = reader.readtext(image)
    text_list = [text for _, text, _ in result]  # Extract text component
    return text_list


def preprocessing_pipeline(image_path):
    """Full pipeline: load, preprocess, deskew, extract, and clean text."""
    # Load image
    image = load_image(image_path)
    logging.info("Loaded image")
    # Preprocess for skew detection
    thresh = preprocess_image(image)
    logging.info("preprocessed image")
    # Detect lines and calculate skew
    lines = detect_lines(thresh)
    skew_angle = calculate_skew_angle(lines)
    logging.info("Calculated skew")
    # Deskew the image
    deskewed = deskew_image(image, skew_angle)
    logging.info("Deskewed image")
    # Extract raw text
    raw_text = extract_text(deskewed)
    logging.info(f"Extracted raw text: {raw_text}")
    # Clean and reconstruct text
    cleaned_text = clean_text(raw_text)
    logging.info(f"cleaned text: {cleaned_text}")
    # Check and correct text using LLaVA
    cleaned_text = check_text_via_llava(cleaned_text)
    logging.info("cleaned text via llava")
    return cleaned_text


def check_text_via_llava(text, model="mistral", host_url=OLLAMA_HOST_URL):
    """Checks and corrects the give text using the LLaVA model runing on Ollama and returns the response."""

    prompt = f"""
    Clean and correct the following text while ensuring it makes sense. Remove any gibberish, repeated characters, or nonsensical parts. Only return the corrected text without any explanations.  

    Text: "{text}"
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(host_url, json=payload)
    logging.info("Got response")
    result = response.json()
    cleaned_text = result["response"].strip().lower()
    return cleaned_text


# Example usage
if __name__ == "__main__":
    image_path = os.path.join(os.getcwd(), "received_images", "text.jpg")
    cleaned_text = preprocessing_pipeline(image_path)
    # print("Cleaned Extracted Text:")
    print(cleaned_text)
