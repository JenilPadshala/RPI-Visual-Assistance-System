import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import logging
import subprocess
from gtts import gTTS
import threading


# Configure logging
logging.basicConfig(
    filename="sample.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def text_to_speech(text, language='en', slow=False):
    """
    Convert text to speech and play it asynchronously.

    Parameters:
    - text (str): The text to convert to speech.
    - language (str): Language code (default is 'en' for English).
    - slow (bool): Whether to slow down the speech (default is False).
    """
    try:
        speech = gTTS(text=text, lang=language, slow=slow)
        audio_file = "output.mp3"
        speech.save(audio_file)

        # Play audio in a subprocess (non-blocking)
        subprocess.Popen(["mpg321", "-q", audio_file])

        # Remove the file after some delay to avoid clutter
        threading.Timer(2.0, lambda: os.remove(audio_file)).start()
    except Exception as e:
        logging.error(f"Error in text_to_speech: {e}")

def speak_out(text):
    """Run text-to-speech in a separate thread."""
    tts_thread = threading.Thread(target=text_to_speech, args=(text,))
    tts_thread.daemon = True  # Auto-exits when program ends
    tts_thread.start()

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)

from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.hazardous_objects = {"person", "keyboard", "remote"}  # Define hazardous objects inside the class
        self.reported_hazards = set()  # Store track IDs of reported hazards

    def get_object_position(self, bbox, frame_width):
        """
        Determine the position of an object in the frame relative to the user.
        Args:
            bbox (tuple): Bounding box (x_min, y_min, x_max, y_max)
            frame_width (int): Width of the video frame
        Returns:
            str: "Left", "Right", or "Front"
        """
        xmin = bbox.xmin()
        xmax = bbox.xmax()
        normalized_center_x = (xmin + xmax) / 2
        center_x = int(normalized_center_x * frame_width)
        if center_x < frame_width / 3:
            return "Right"
        elif center_x > 2 * frame_width / 3:
            return "Right"
        else:
            return "Left"
        


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        # Get tracking ID
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        track_id = track[0].get_id() if len(track) == 1 else -1
        position = user_data.get_object_position(bbox, width)
        # Check if the object is hazardous and has not been reported yet
        if label in user_data.hazardous_objects and track_id not in user_data.reported_hazards:
            user_data.reported_hazards.add(track_id)  # Mark as reported
            # Get the position of the object
            
            # Log the hazardous object detection with position
            logging.info(f"Hazardous Object Detected: Label: {label} Confidence: {confidence:.2f} Track_id: {track_id} Position: {position}")
            instruction = f"{label} detected towards your {position} side"
            speak_out(instruction)

        string_to_print += (f"Detection: Label: {label} Confidence: {confidence:.2f} Track_id: {track_id} Position: {position}\n")
        # if label == "person":
        #     # Get track ID
        #     track_id = 0
        #     track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        #     if len(track) == 1:
        #         track_id = track[0].get_id()
        #     string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
        #     detection_count += 1
    # if user_data.use_frame:
    #     # Note: using imshow will not work here, as the callback function is not running in the main thread
    #     # Let's print the detection count to the frame
    #     cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     # Example of how to use the new_variable and new_function from the user_data
    #     # Let's print the new_variable and the result of the new_function to the frame
    #     cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     # Convert the frame to BGR
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
