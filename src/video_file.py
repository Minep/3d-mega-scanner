from scanner.Pipeline import PipelineStageBase, Pipeline

import cv2
import time
import numpy as np

class OpenCVFileWrap(PipelineStageBase):
    def __init__(self, file, pipeline: Pipeline) -> None:
        self.__wrapped = pipeline
        self.__file = file

    def initialize(self, options):
        super().initialize(options)
        self.__capture = cv2.VideoCapture(self.__file)
        self.__wrapped.initialize(options)

    def execute(self, in_obj=None):
        if not self.__capture.isOpened():
            raise Exception("VideoCapture is not opened")
        t = 1 / self._opts.fps
        while True:
            ret, frame = self.__capture.read()
            if not ret:
                print("Warn: empty frame!")
            else:
                frame = cv2.resize(frame, (1280, 720))
                result = self.__wrapped.execute(frame)
                result = result / result.max() * 255
                cv2.imshow('frame', result.astype(np.uint8))
            if cv2.waitKey(1) == ord('q'):
                break
            time.sleep(t)
