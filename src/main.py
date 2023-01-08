from scanner.Pipeline import Pipeline, PipelineStageBase, PipelineOptions
from scanner.depth_extract.Extractor import DepthExtractLeReS
from scanner.depth_extract.Visualize import PointCloudVisualizer, PointCloudVisualizerRT

from v4l2 import GstV4L2CaptureWrap
from video_file import OpenCVFileWrap
from options import global_options

import cv2

class CV2ReadImage(PipelineStageBase):
    def __init__(self) -> None:
        super().__init__("image_input (cv2)")
    def initialize(self, options):
        self.__opts = options

    def execute(self, in_obj):
        return cv2.imread(in_obj)

def main():
    options = PipelineOptions(global_options)
    pipeline = Pipeline([
        CV2ReadImage(),
        DepthExtractLeReS(),
        PointCloudVisualizer()
    ])

    pipeline.initialize(options)
    pipeline.execute("../images/img1.jpg")

def main_seq():
    options = PipelineOptions(global_options)
    pipeline = GstV4L2CaptureWrap(Pipeline([
        DepthExtractLeReS(),
        #PointCloudVisualizerRT()
    ]))

    pipeline.initialize(options)
    pipeline.execute()

def main_seq_file():
    options = PipelineOptions(global_options)
    pipeline = OpenCVFileWrap("../images/vid1.mp4",Pipeline([
        DepthExtractLeReS()
    ]))

    pipeline.initialize(options)
    pipeline.execute()


if __name__ == "__main__":
    main_seq()
