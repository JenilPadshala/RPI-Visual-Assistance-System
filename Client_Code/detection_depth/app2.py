import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
import numpy as np
import pathlib
import hailo
from hailo_apps_infra.hailo_rpi_common import app_callback_class, get_caps_from_pad
from pipeline import GStreamerDetectionDepthApp
import logging
from collections import deque
import json
import os
import subprocess

logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

HAZARDOUS_OBJECTS = {"person", "bed", "laptop"}

# --- Constants for Direction Detection ---
DIRECTION_LEFT_THRESHOLD = 0.35  # Object center < 35% of width is LEFT
DIRECTION_RIGHT_THRESHOLD = 0.65  # Object center > 65% of width is RIGHT
# Object center between these thresholds is FRONT
# -----------------------------------------

# --- Constants for Motion Detection ---
MOTION_DEPTH_HISTORY = 5  # Number of depth values to consider
MOTION_MIN_SAMPLES = 2  # Minimum depth values needed to determine motion
# MOTION_DEPTH_DIFF_THRESHOLD = 0.05 # Minimum difference (in meters?) to consider as actual motion
MOTION_SLOPE_THRESHOLD = (
    0.07  # Min slope magnitude (depth units per frame index) to consider as motion
)
# ------------------------------------


class UserData(app_callback_class):
    def __init__(self):
        super().__init__()

        self.tracked_hazards = {}
        # Structure per track_id:
        # {
        #     "label": str,
        #     "latest_center": (float, float), # (cx, cy)
        #     "center_depths": deque(maxlen=MOTION_DEPTH_HISTORY), # Stores recent valid depths
        #     "motion": str, # "Towards", "Away", "Stationary", "Undetermined"
        #     "direction": str # "Left", "Right", "Front", "Undetermined"
        # }
        # -----------------------------------------

        # --- Buffer for recent depth maps (for center depth calculation) ---
        self.depth_map_buffer = {}  # Stores {frame_num: numpy_depth_array}
        self.depth_map_buffer_size = 5  # Store last 5 depth maps
        self.last_processed_depth_frame = -1  # Prevent redundant processing
        self.announced_hazards = set()
        # -------------------------------------------------------------------------

    # def calculate_average_depth(self, depth_mat):
    #     depth_values = np.array(depth_mat).flatten()  # Flatten the array and filter out outlier pixels
    #     try:
    #         m_depth_values = depth_values[depth_values <= np.percentile(depth_values, 95)]  # drop 5% of highest values (outliers)
    #     except Exception as e:
    #         m_depth_values = np.array([])
    #     if len(m_depth_values) > 0:
    #         average_depth = np.mean(m_depth_values)  # Calculate the average depth of the pixels
    #     else:
    #         average_depth = 0  # Default value if no valid pixels are found
    #     logging.info("calculated average depth")
    #     return average_depth


def app_callback_detection(pad, info, user_data):
    logging.info("Entered app_callback_detection")
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    caps = get_caps_from_pad(pad)  # Use the helper function

    det_width = int(caps[1])
    det_height = int(caps[2])

    user_data.increment()
    frame_num = user_data.frame_count
    # print(f"Detection frame_num: {frame_num}")

    roi = hailo.get_roi_from_buffer(buffer)
    if roi is None:
        return Gst.PadProbeReturn.OK

    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    if detections:
        logging.info(f"-- Frame {frame_num} --")
        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            bbox = detection.get_bbox()
            track_id = -1
            track_objs = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if track_objs:
                track_id = track_objs[0].get_id()

            is_hazard = label in HAZARDOUS_OBJECTS
            has_valid_id = track_id != -1

            if is_hazard and has_valid_id:
                if track_id not in user_data.tracked_hazards:
                    user_data.tracked_hazards[track_id] = {
                        "label": label,
                        "latest_center": None,
                        "center_depths": deque(maxlen=MOTION_DEPTH_HISTORY),
                        "motion": "Undetermined",
                        "direction": "Undetermined",
                    }
                    logging.info(
                        f"Started tracking new hazardous object: Label='{label}', TrackID={track_id}"
                    )

                hazard_data = user_data.tracked_hazards[track_id]
                # ---------------------------------------------------------

                # --- Calculate Current Center Coordinates ---
                cx = bbox.xmin() + bbox.width() / 2
                cy = bbox.ymin() + bbox.height() / 2
                latest_center = (round(cx, 2), round(cy, 2))
                hazard_data["latest_center"] = latest_center  # Update latest center
                # -----------------------------------------

                # --- Determine Direction (Left/Right/Front) ---
                relative_x = cx
                if relative_x < DIRECTION_LEFT_THRESHOLD:
                    hazard_data["direction"] = "Left"
                elif relative_x > DIRECTION_RIGHT_THRESHOLD:
                    hazard_data["direction"] = "Right"
                else:
                    hazard_data["direction"] = "Front"
                # ---------------------------------------------

                # --- Calculate Current Center Depth ---
                current_center_depth = None
                depth_data = user_data.depth_map_buffer.get(frame_num - 1)

                logging.info(depth_data)

                if depth_data is not None:
                    try:
                        depth_h, depth_w = depth_data.shape
                        if depth_w > 0 and depth_h > 0:  # Check depth map dimensions
                            scaling_factor_x = depth_w / det_width
                            scaling_factor_y = depth_h / det_height
                            scaled_cx = cx * scaling_factor_x
                            scaled_cy = cy * scaling_factor_y
                            depth_x = max(0, min(int(round(scaled_cx)), depth_w - 1))
                            depth_y = max(0, min(int(round(scaled_cy)), depth_h - 1))
                            current_center_depth = float(
                                depth_data[depth_y, depth_x]
                            )  # Get depth
                            # Append valid depth to history
                            hazard_data["center_depths"].append(
                                round(current_center_depth, 2)
                            )
                        else:
                            logging.warning(
                                f"Invalid depth map dimensions ({depth_w}x{depth_h}) for frame {frame_num}"
                            )

                    except Exception as e:
                        logging.error(
                            f"Error calculating center depth for hazard ID {track_id}, Frame {frame_num}: {e}"
                        )
                        current_center_depth = None  # Ensure it remains None on error
                # ------------------------------------

                # # --- Determine Motion (Towards/Away/Stationary) ---
                # depth_history = hazard_data["center_depths"]
                # if len(depth_history) >= MOTION_MIN_SAMPLES:
                #     # Compare latest depth to the one before it
                #     latest_depth = depth_history[-1]
                #     previous_depth = depth_history[-2]

                #     depth_diff = latest_depth - previous_depth

                #     if abs(depth_diff) > MOTION_DEPTH_DIFF_THRESHOLD:
                #         if depth_diff < 0:
                #             hazard_data["motion"] = "Towards"
                #         else:
                #             hazard_data["motion"] = "Away"
                #     else:
                #         hazard_data["motion"] = "Stationary"
                # # elif len(depth_history) > 0:
                #     #  Not enough history yet, but has at least one point
                #     #  hazard_data["motion"] = "Stationary" # Assume stationary initially
                # else:
                #      # No valid depth history at all
                #      hazard_data["motion"] = "Undetermined"
                # # -------------------------------------------------

                # --- Determine Motion using Linear Regression ---
                depth_history = hazard_data["center_depths"]
                if len(depth_history) >= MOTION_MIN_SAMPLES:
                    try:
                        # Prepare data for regression
                        y_values = np.array(depth_history)
                        x_values = np.arange(len(y_values))

                        # Perform linear regression
                        slope, intercept = np.polyfit(x_values, y_values, 1)

                        # Determine motion based on the slope
                        if slope < -MOTION_SLOPE_THRESHOLD:
                            hazard_data["motion"] = "Towards"
                        elif slope > MOTION_SLOPE_THRESHOLD:
                            hazard_data["motion"] = "Away"
                        else:
                            hazard_data["motion"] = "Stationary"
                        # logging.debug(f"[Frame {frame_num}, ID {track_id}] Motion Check: History={list(depth_history)}, Slope={slope:.3f}, Motion={hazard_data['motion']}")

                    except Exception as e:
                        logging.error(
                            f"[Frame {frame_num}, ID {track_id}] Error during motion regression: {e}"
                        )
                        hazard_data["motion"] = "Undetermined"  # Fallback on error
                else:
                    # Not enough data points for reliable regression
                    hazard_data["motion"] = "Undetermined"
                # -------------------------------------------------

                # --- Trigger TTS Announcement via Subprocess ---
                label = hazard_data.get("label")
                motion = hazard_data.get("motion")
                direction = hazard_data.get("direction")

                if (
                    label
                    and motion
                    and motion != "Undetermined"
                    and direction
                    and direction != "Undetermined"
                    and track_id not in user_data.announced_hazards
                ):
                    # Construct the announcement message
                    if motion == "Stationary":
                        if direction == "Front":
                            announcement = f"{label} is stationary in front of you."
                        announcement = f"{label} is {motion.lower()} on your {direction.lower()} side."
                    elif motion == "Towards":
                        if direction == "Front":
                            announcement = (
                                f"{label} is moving towards you in front of you."
                            )
                        else:
                            announcement = f"{label} is moving towards you on your {direction.lower()} side."
                    elif motion == "Away":
                        if direction == "Front":
                            announcement = (
                                f"{label} is moving away from you in front of you."
                            )
                        else:
                            announcement = f"{label} is moving away from you on your {direction.lower()} side."
                    else:
                        announcement = f"{label} is on your {direction.lower()} side, motion {motion.lower()}."

                    logging.info(
                        f"Dispatching TTS for Track ID {track_id}: '{announcement}'"
                    )

                    # --- Call tts_handler.py ---
                    try:
                        app_dir = pathlib.Path(__file__).parent.resolve()
                        tts_script_path = app_dir / "tts_handler.py"
                        command = [
                            "python3",
                            str(tts_script_path),
                            "--text",
                            announcement,
                        ]
                        process = subprocess.Popen(command)
                        logging.info(
                            f"Launched TTS handler process (PID: {process.pid}) for Track ID {track_id}"
                        )
                        # Marking this track ID as announced immediately after launching
                        user_data.announced_hazards.add(track_id)
                        logging.info(f"Track ID {track_id} marked as announced.")
                    except FileNotFoundError:
                        logging.error(
                            f"TTS Error: 'python3' or '{tts_script_path}' not found. Cannot launch TTS handler."
                        )
                        print(
                            f"TTS Error: Could not find python3 or tts_handler.py. Please ensure both are available."
                        )
                    except Exception as e:
                        logging.error(
                            f"TTS Error: Failed to launch TTS handler subprocess for Track ID {track_id}: {e}"
                        )
                        print(
                            f"Error: Failed to launch TTS subprocess. Check system.log."
                        )
                    # --- End Subprocess Call ---

                # --- End TTS Section ---

    return Gst.PadProbeReturn.OK


def app_callback_depth(pad, info, user_data):
    logging.info("Entered app_callback_depth")
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    frame_num_depth = user_data.frame_count
    if frame_num_depth <= user_data.last_processed_depth_frame:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    if roi is None:
        return Gst.PadProbeReturn.OK

    depth_mats = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)

    if depth_mats:
        depth_mat_obj = depth_mats[0]
        depth_data_raw = depth_mat_obj.get_data()
        height = depth_mat_obj.get_height()
        width = depth_mat_obj.get_width()

        if height > 0 and width > 0:
            # --- Store depth map in buffer ---
            try:
                depth_data_np = np.array(depth_data_raw).reshape((height, width))
                user_data.depth_map_buffer[frame_num_depth] = depth_data_np
                # print(user_data.depth_map_buffer)
                print(user_data.depth_map_buffer[frame_num_depth].size)
                # print(len(user_data.depth_map_buffer))
                if len(user_data.depth_map_buffer) > user_data.depth_map_buffer_size:
                    oldest_frame = min(user_data.depth_map_buffer.keys())
                    del user_data.depth_map_buffer[oldest_frame]
            except Exception as e:
                logging.error(
                    f"Error processing/storing depth map for frame {frame_num_depth}: {e}"
                )
            # --------------------------------
        else:
            logging.warning(
                f"[Depth Frame ~{frame_num_depth}] Received depth map with invalid dimensions: {width}x{height}"
            )
    return Gst.PadProbeReturn.OK


# --- Main execution block ---
if __name__ == "__main__":
    logging.info("main: Initializing application...")
    user_data_obj = UserData()
    app_dir = pathlib.Path(__file__).parent.resolve()
    logging.info(f"Application directory: {app_dir}")

    app = GStreamerDetectionDepthApp(
        app_callback_det=app_callback_detection,
        app_callback_dep=app_callback_depth,
        user_data=user_data_obj,
        app_path=app_dir,
        parser=None,
    )

    logging.info("main: Starting GStreamer pipeline run...")
    try:
        app.run()
    except KeyboardInterrupt:
        logging.info("main: Keyboard interrupt received. Stopping pipeline...")
    except Exception as e:
        logging.exception(f"main: An unexpected error occurred during app.run(): {e}")
    finally:
        logging.info("main: GStreamer pipeline finished or interrupted.")

        # --- Print the final tracked hazardous object information ---
        logging.info("=" * 60)
        logging.info("Final Summary of Tracked Hazardous Objects:")
        if user_data_obj.tracked_hazards:
            # Log structure for readability
            for track_id, hazard_data in user_data_obj.tracked_hazards.items():
                logging.info(
                    f"  --- Track ID: {track_id} (Label: {hazard_data.get('label', 'N/A')}) ---"
                )
                logging.info(
                    f"    Latest Center (Cx, Cy): {hazard_data.get('latest_center', 'N/A')}"
                )
                logging.info(f"    Direction: {hazard_data.get('direction', 'N/A')}")
                logging.info(f"    Motion: {hazard_data.get('motion', 'N/A')}")
                depth_list = list(hazard_data.get("center_depths", []))
                logging.info(f"    Last {len(depth_list)} Center Depths: {depth_list}")

            # Print as JSON to standard output
            try:
                # Convert deques to lists for JSON serialization
                serializable_data = {}
                for track_id, data in user_data_obj.tracked_hazards.items():
                    serializable_data[track_id] = {
                        "label": data.get("label"),
                        "latest_center": data.get("latest_center"),
                        "direction": data.get("direction"),
                        "motion": data.get("motion"),
                        "center_depths": list(data.get("center_depths", [])),
                    }

                print("\n--- JSON Summary of Tracked Hazards ---")
                print(json.dumps(serializable_data, indent=2))
                print("---------------------------------------\n")
            except Exception as e:
                logging.error(f"Could not serialize tracked hazards to JSON: {e}")

        else:
            logging.info("  No hazardous objects were tracked during this run.")
        logging.info("=" * 60)
        # -------------------------------------------------------------

        logging.info("main: Exiting application.")
# --- End of script --
