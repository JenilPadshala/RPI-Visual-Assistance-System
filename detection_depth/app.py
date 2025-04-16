# import gi
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst
# import numpy as np
# import pathlib
# import hailo
# from hailo_apps_infra.hailo_rpi_common import app_callback_class, get_caps_from_pad
# from pipeline import GStreamerDetectionDepthApp
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import pathlib
import hailo
from hailo_apps_infra.hailo_rpi_common import app_callback_class, get_caps_from_pad
from pipeline import GStreamerDetectionDepthApp
import logging

logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class UserData(app_callback_class):
    def __init__(self):
        super().__init__()
    
    def calculate_average_depth(self, depth_mat):
        depth_values = np.array(depth_mat).flatten()  # Flatten the array and filter out outlier pixels
        try:
            m_depth_values = depth_values[depth_values <= np.percentile(depth_values, 95)]  # drop 5% of highest values (outliers)          
        except Exception as e:
            m_depth_values = np.array([])
        if len(m_depth_values) > 0:
            average_depth = np.mean(m_depth_values)  # Calculate the average depth of the pixels
        else:
            average_depth = 0  # Default value if no valid pixels are found
        logging.info("calculated average depth")
        return average_depth
        

def app_callback_detection(pad, info, user_data):
    logging.info("Entered app_callback_detection")
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    caps = get_caps_from_pad(pad) # Use the helper function
    print(caps)

    frame_num = user_data.increment()
    roi = hailo.get_roi_from_buffer(buffer)
    if roi is None:
        return Gst.PadProbeReturn.OK
    
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    if detections:
        logging.info(f"-- Frame {frame_num} --")
        detection_details = []
        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            bbox = detection.get_bbox()

            track_id = -1
            track_objs = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if track_objs:
                track_id = track_objs[0].get_id()
            
            detection_details.append(
                f"  Detection: {label} (Conf: {confidence:.2f}, TrackID: {track_id}, "
                f"Box: ({bbox.xmin():.2f}, {bbox.ymin():.2f}, {bbox.width():.2f}, {bbox.height():.2f}))"
            )
        if detection_details:
            logging.info("[Detection Details]")
            logging.info("\n".join(detection_details))   
        
    return Gst.PadProbeReturn.OK

def app_callback_depth(pad, info, user_data):
    logging.info("Entered app_callback_depth")
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    frame_num = user_data.increment()
    roi = hailo.get_roi_from_buffer(buffer)
    if roi is None:
        return Gst.PadProbeReturn.OK
    depth_mats = roi.get_objects_typed(hailo.HAILO_DEPTH_MASK)
    if depth_mats:
        print(len(depth_mats))
        # Assume one depth map per buffer from this branch
        depth_mat_obj = depth_mats[0] 
        depth_data = depth_mat_obj.get_data() 
        height = depth_mat_obj.get_height()
        width = depth_mat_obj.get_width()
        
        if height > 0 and width > 0:
            avg_depth = user_data.calculate_average_depth(depth_data)            
            # logging.info depth info separately
            print(f"Depth Map: {width}x{height}")
            logging.info(f"[Depth Branch Data - Frame {frame_num}]") # Frame num might differ slightly if branches run at different speeds
            logging.info(f"  Depth Map: {width}x{height}, Avg Depth: {avg_depth:.2f}, Center Depth: {avg_depth:.2f}")
        else:
             logging.info(f"[Depth Branch Data - Frame {frame_num}]")
             logging.info("  Warning: Received depth map with invalid dimensions.")

   

    return Gst.PadProbeReturn.OK   



if __name__ == "__main__":
    logging.info("main")
    user_data_obj = UserData()
    app = GStreamerDetectionDepthApp(app_callback_detection, app_callback_depth, user_data_obj, pathlib.Path(__file__).parent.resolve())
    logging.info("starting run")
    app.run()
    logging.info("Exiting...")