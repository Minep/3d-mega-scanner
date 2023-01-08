from scanner.Pipeline import PipelineStageBase, Pipeline

import cv2
import time
import numpy as np

class GstV4L2CaptureWrap(PipelineStageBase):
    def __init__(self, pipeline: Pipeline) -> None:
        self.__wrapped = pipeline
        self.__gst_pipeline = 'v4l2src device={} ! videoconvert ! videorate ! video/x-raw, framerate={}/1 ! appsink drop=1'

    def initialize(self, options):
        super().initialize(options)
        gst = self.__gst_pipeline.format(options.capture_dev, options.fps)
        print("Starting Gst pipeline using:\n\t", gst)
        self.__capture = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        self.__wrapped.initialize(options)

    def execute(self, in_obj=None):
        if not self.__capture.isOpened():
            raise Exception("VideoCapture is not opened")
        t = 1 / self._opts.fps
        try:
            while True:
                ret, frame = self.__capture.read()
                if not ret:
                    print("Warn: empty frame!")
                else:
                    r = self.__wrapped.execute(frame)
                    cv2.imshow("frame", r)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(t)

        except KeyboardInterrupt:
            print("terminated")
            pass
        self.__wrapped.cleanup()
