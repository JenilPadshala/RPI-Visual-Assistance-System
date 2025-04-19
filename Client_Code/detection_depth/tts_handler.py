# tts_handler.py
import argparse
import subprocess
import os
import logging
from gtts import gTTS

logging.basicConfig(
    filename="tts_handler.log", # Log TTS specific messages here
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def text_to_speech(text, language='en', slow=False):
    """
    Convert text to speech and play it using mpg321.

    Parameters:
    - text (str): The text to convert to speech.
    - language (str): Language code (default is 'en' for English).
    - slow (bool): Whether to slow down the speech (default is False).

    Note: Requires 'gTTS' library and 'mpg321' command-line tool.
    """
    audio_file = None # Initialize to ensure it's defined for finally block
    try:
        if not text:
            logging.warning("Received empty text for TTS. Skipping.")
            return

        # Create gTTS object
        speech = gTTS(text=text, lang=language, slow=slow)

        # Define audio file name (use PID for potential parallel runs, more robust)
        audio_file = f"output_tts_{os.getpid()}.mp3"

        # Save the audio file
        speech.save(audio_file)
        logging.info(f"TTS audio saved to {audio_file} for text: '{text}'")

        # Play the audio file using mpg321 (ensure it's installed)
        # -q suppresses mpg321's own output
        process = subprocess.run(["mpg321", "-q", audio_file], check=True, capture_output=True, text=True)
        logging.info(f"TTS audio played successfully for {audio_file}. mpg321 stderr: {process.stderr.strip()}")

    except FileNotFoundError:
        logging.error("TTS Error: 'mpg321' command not found. Please install it (e.g., 'sudo apt install mpg321').")
        print("TTS Error: 'mpg321' command not found. Please install it.") # Also print to console
    except subprocess.CalledProcessError as e:
         logging.error(f"TTS Error: 'mpg321' failed with exit code {e.returncode}. Stderr: {e.stderr.strip()}")
         print(f"TTS Error: 'mpg321' failed. Check tts_handler.log for details.")
    except Exception as e:
        logging.error(f"TTS Error: Failed to generate or play speech for text '{text}': {e}")
        print(f"TTS Error: Failed to generate or play speech. Check tts_handler.log.")
    finally:
        # Remove the file after playing or if an error occurred
        if audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                logging.info(f"Temporary TTS file {audio_file} removed.")
            except OSError as e:
                logging.error(f"Error removing temporary TTS file {audio_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Speech Handler")
    parser.add_argument("--text", required=True, help="The text string to convert to speech.")
    parser.add_argument("--lang", default="en", help="Language code for TTS (e.g., 'en', 'es').")
    parser.add_argument("--slow", action="store_true", help="Use slower speech.")

    args = parser.parse_args()

    logging.info(f"TTS Handler called with text: '{args.text}'")
    print(f"TTS Handler received: '{args.text}'") # Optional: confirm reception

    text_to_speech(text=args.text, language=args.lang, slow=args.slow)

    logging.info("TTS Handler finished.")
    # print("TTS Handler finished.") # Optional: confirm exit