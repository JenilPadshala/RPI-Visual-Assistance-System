import cv2
import numpy as np
import easyocr


def load_image(image_path):
    """
    Load the image from the given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image


def preprocess_image(image):
    """
    Preprocess the image: convert to grayscale, apply Gaussian blur, and adaptive thresholding.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Thresholded image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def detect_lines(thresh):
    """
    Detect lines in the thresholded image using Hough Transform.

    Args:
        thresh (np.ndarray): Thresholded image.

    Returns:
        list: List of detected lines.
    """
    lines = cv2.HoughLinesP(
        thresh, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )
    return lines


def calculate_skew_angle(lines):
    """
    Calculate the skew angle based on detected lines.

    Args:
        lines (list): List of detected lines.

    Returns:
        float: Skew angle in degrees.
    """
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
    """
    Deskew the image based on the calculated skew angle.

    Args:
        image (np.ndarray): Original image.
        skew_angle (float): Skew angle in degrees.

    Returns:
        np.ndarray: Deskewed image.
    """
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


def extract_text(image):
    """
    Extract text from the image using EasyOCR.

    Args:
        image (np.ndarray): Input image.

    Returns:
        list: List of extracted text.
    """
    reader = easyocr.Reader(["en"])  # Initialize EasyOCR reader for English
    result = reader.readtext(image)
    text_list = [text for _, text, _ in result]  # Extract only the text component
    return text_list


def preprocessing_pipeline(image_path):
    """
    Complete preprocessing pipeline: load, preprocess, deskew, and extract text.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: Extracted text from the deskewed image.
    """
    # Load image
    image = load_image(image_path)

    # Preprocess image
    thresh = preprocess_image(image)

    # Detect lines
    lines = detect_lines(thresh)

    # Calculate skew angle
    skew_angle = calculate_skew_angle(lines)

    # Deskew image
    deskewed = deskew_image(image, skew_angle)

    # Extract text using EasyOCR
    text = extract_text(deskewed)

    return text


# Example usage
if __name__ == "__main__":
    image_path = ""
    extracted_text = preprocessing_pipeline(image_path)
    for text in extracted_text:
        print(text)
