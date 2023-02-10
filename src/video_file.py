from scanner.Pipeline import PipelineStageBase, Pipeline

import cv2
import time
import numpy as np

class OpenCVFileWrap(PipelineStageBase):
    def __init__(self, file, pipeline: Pipeline) -> None:
        self.__wrapped = pipeline
        self.__file = file
        self.__gst_pipeline = 'filesrc location={} ! decodebin ! videoconvert ! videorate ! video/x-raw, framerate={}/1 ! appsink'

    def initialize(self, options):
        super().initialize(options)
        gst = self.__gst_pipeline.format(self.__file, options.fps)
        print("Starting Gst pipeline using:\n\t", gst)
        self.__capture = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        self.__wrapped.initialize(options)

    def execute(self, in_obj=None):
        if not self.__capture.isOpened():
            raise Exception("VideoCapture is not opened")
        t = 1 / 32
        while True:
            ret, frame = self.__capture.read()
            if ret:
                frame = cv2.resize(frame, (1280, 720))
                result = self.__wrapped.execute(frame)
                if result is not None:
                    result = result[0]
                    result = result / result.max() * 255
                    cv2.imshow('frame', result.astype(np.uint8))
            if cv2.waitKey(1) == ord('q'):
                break
            self.__wrapped.tick()
            time.sleep(t)

