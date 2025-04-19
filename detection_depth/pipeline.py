import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
import setproctitle
from hailo_apps_infra.hailo_rpi_common import detect_hailo_arch, get_default_parser
from hailo_apps_infra.gstreamer_helper_pipelines import DISPLAY_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, SOURCE_PIPELINE, TRACKER_PIPELINE, USER_CALLBACK_PIPELINE
from hailo_apps_infra.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback
import logging

logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# User Gstreamer Application: This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionDepthApp(GStreamerApp):
    def __init__(self, app_callback_det,app_callback_dep, user_data, app_path, parser=None):
        if parser == None:
            parser = get_default_parser()
        parser.add_argument('--apps_infra_path', default='None', help='Required argument. Path to the hailo-apps-infra folder.')
        super().__init__(parser, user_data)  # Call the parent class constructor


        self.app_callback_detection_func = app_callback_det
        self.app_callback_depth_func = app_callback_dep


        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError('Could not auto-detect Hailo architecture. Please specify --arch manually.')
            self.arch = detected_arch
        else:
            self.arch = self.options_menu.arch
            print(f'Using Hailo architecture: {self.arch}')

        if self.options_menu.apps_infra_path is None:
            raise ValueError('Please specify path to the hailo-apps-infra folder')
        elif not os.path.exists(self.options_menu.apps_infra_path):
            raise ValueError('Please specify valid path to the hailo-apps-infra folder')

        self.app_callback = app_callback_det
        setproctitle.setproctitle("Hailo Detection Cropper App")  # Set the process title

        # Set Hailo parameters (for detection neural network) these parameters should be set based on the model used
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45        
        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the HEF file path & depth post processing method name based on the arch
        if self.arch == "hailo8":
            self.detection_hef_path = self.options_menu.apps_infra_path + '/resources/yolov11n.hef'
            self.depth_hef_path = self.options_menu.apps_infra_path + '/resources/scdepthv3.hef'
        else:  # hailo8l
            self.detection_hef_path = self.options_menu.apps_infra_path + '/resources/yolov8s_h8l.hef'
            self.depth_hef_path = self.options_menu.apps_infra_path + '/resources/scdepthv3_h8l.hef'
        self.depth_post_function_name = "filter_scdepth"

        # Set the post-processing shared object file
        self.detection_post_process_so = self.options_menu.apps_infra_path + '/resources/libyolo_hailortpp_postprocess.so'
        self.detection_post_function_name = "filter_letterbox"
        self.depth_post_process_so = self.options_menu.apps_infra_path + '/resources/libdepth_postprocess.so'
        self.depth_post_function_name = "filter_scdepth"

        self.create_pipeline()
        logging.info("done with init in pipeline")

        self._connect_specific_probes()

    def _connect_specific_probes(self):
        if self.pipeline is None:
            logging.error("Pipeline not created before connecting probes.")
            return
        
        probe_connected = False

        #Connect detection callback
        element_det = self.pipeline.get_by_name("branch_1_user_callback")
        if element_det:
            pad_det = element_det.get_static_pad("src")
            if pad_det and not self.options_menu.disable_callback:
                pad_det.add_probe(Gst.PadProbeType.BUFFER, self.app_callback_detection_func, self.user_data)
                probe_connected = True
            elif not pad_det:
                logging.warning("Could not get pad for detection callback element.")
        else:
            logging.warning("Detection callback element 'branch_1_user_callback' not found")
        
        # Connect depth callback
        element_dep = self.pipeline.get_by_name("branch_2_user_callback")
        if element_dep:
            pad_dep = element_dep.get_static_pad("src") # Or sink pad
            if pad_dep and not self.options_menu.disable_callback:
                pad_dep.add_probe(Gst.PadProbeType.BUFFER, self.app_callback_depth_func, self.user_data)
                logging.info(f"Connected depth probe to '{element_dep.get_name()}' {pad_dep.get_name()} pad.")
                probe_connected = True
            elif not pad_dep:
                 logging.warning("Could not get pad for depth callback element.")
        else:
            logging.warning("Depth callback element 'branch_2_user_callback' not found.")

        if not probe_connected and not self.options_menu.disable_callback:
             logging.error("Failed to connect probe to any callback element.")

    def get_pipeline_string(self):
        
        target_width = 320
        target_height = 256
        logging.info(f"Setting target pipeline dimensions to: {target_width}x{target_height}")

        source_pipeline = SOURCE_PIPELINE(self.video_source, self.video_width, self.video_height)
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.detection_hef_path,
            post_process_so=self.detection_post_process_so,
            post_function_name=self.detection_post_function_name,
            batch_size=self.batch_size,
            additional_params=self.thresholds_str,
            name ='detection_inference')
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=-1)
        user_callback_pipeline_1 = USER_CALLBACK_PIPELINE(name="branch_1_user_callback")
        # user_callback_pipeline_1 = USER_CALLBACK_PIPELINE()
        display_pipeline_1 = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps, name="detection_display")
        depth_pipeline = INFERENCE_PIPELINE(
            hef_path=self.depth_hef_path,
            post_process_so=self.depth_post_process_so,
            post_function_name=self.depth_post_function_name,
            name='depth_inference')
        depth_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(depth_pipeline, name='inference_wrapper_depth')
        user_callback_pipeline_2 = USER_CALLBACK_PIPELINE(name="branch_2_user_callback")
        display_pipeline_2 = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps, name="depth_display")

        scaler_pipeline = (
            f'videoconvert ! videoscale ! '
            f'video/x-raw,width={target_width},height={target_height} ! '
            f'videoconvert'
        )

        logging.info("got pipeline_strings")
        return (
            f'{source_pipeline} ! '
            f'{scaler_pipeline} ! '  # Resize the stream here
            f'tee name=t '
            # branch 1
            f't. ! queue ! {detection_pipeline_wrapper} ! {tracker_pipeline} ! {user_callback_pipeline_1} ! {display_pipeline_1} '
            # branch 2
            f't. ! queue ! {depth_pipeline_wrapper} ! {user_callback_pipeline_2} ! {display_pipeline_2}'
        )

if __name__ == '__main__':
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionDepthApp(app_callback, user_data, app_path='/home/pi/hailo-rpi5-examples/hailo-apps-infra/', parser=None)
    logging.info("main of pipeline")
    app.run()